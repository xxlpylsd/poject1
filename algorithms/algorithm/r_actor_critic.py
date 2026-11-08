"""
# @Time    : 2021/7/1 6:53 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : r_actor_critic.py
"""

import torch
import torch.nn as nn
from algorithms.utils.util import init, check
from algorithms.utils.cnn import CNNBase
from algorithms.utils.mlp import MLPBase
from algorithms.utils.rnn import RNNLayer
from algorithms.utils.act import (
    ACTLayer,
    FlatJointACTLayer,
    GatedJointACTLayer,
    PaperFactorizedACTLayer,
    RelationalSubsetACTLayer,
    StructuredSubsetACTLayer,
)
from algorithms.utils.popart import PopArt
from algorithms.utils.transformer import TransformerBase, count_parameters
from algorithms.utils.transformer import remap_legacy_state_dict
from utils.util import get_shape_from_obs_space


def _make_base(args, obs_shape, role, hidden_override=None):
    if hidden_override is not None and int(hidden_override) > 0:
        import copy as _copy
        args = _copy.copy(args)
        args.hidden_size = int(hidden_override)
    backbone = str(getattr(args, f"{role}_backbone", "mlp")).lower()
    if backbone == "transformer":
        token_dim = int(getattr(args, f"{role}_token_dim", 1) or 1)
        return TransformerBase(args, obs_shape, token_dim=token_dim)
    if len(obs_shape) == 3:
        return CNNBase(args, obs_shape)
    return MLPBase(
        args,
        obs_shape,
        token_dim=(
            int(getattr(args, f"{role}_token_dim", 1) or 1)
            if (
                role == "actor"
                and bool(getattr(args, "actor_token_norm", False))
            )
            else 1
        ),
    )


def _remap_state(state_dict):
    if any(".encoder." in key for key in state_dict):
        return remap_legacy_state_dict(state_dict)
    return state_dict


class R_Actor(nn.Module):

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.backbone_name = str(getattr(args, "actor_backbone", "mlp")).lower()
        self._actor_token_dim = int(
            getattr(args, "actor_token_dim", 1)
        )
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        self.base = _make_base(args, obs_shape, "actor")

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.joint_head_type = str(
            getattr(args, "joint_action_head", "gated")
        ).lower()
        nvec = getattr(action_space, "nvec", None)
        self._gated_joint = False
        self._structured_subset = False
        self._paper_factorized = False
        self._score_ap_tokens = True
        if (
            self.joint_head_type == "paper_factorized"
            and nvec is not None
            and len(nvec) == 1
        ):
            num_modes = int(getattr(args, "joint_num_modes", 3))
            num_choices = int(
                getattr(args, "joint_num_cluster_choices", 0)
            )
            valid_indices = list(getattr(
                args, "joint_valid_action_indices", []
            ))
            choice_sizes = list(getattr(
                args, "joint_cluster_choice_sizes", []
            ))
            choice_ranks = list(getattr(
                args, "joint_cluster_choice_ranks", []
            ))
            action_choices = list(getattr(
                args, "joint_action_cluster_choice", []
            ))
            if (
                self.backbone_name != "mlp"
                or num_modes != 3
                or num_choices <= 0
                or not valid_indices
                or len(valid_indices) != int(nvec[0])
                or not choice_sizes
                or not choice_ranks
                or len(action_choices) != len(valid_indices)
            ):
                raise ValueError(
                    "paper_factorized requires the MLP actor and complete "
                    "mode/cluster-choice metadata"
                )
            size_options = tuple(
                sorted({int(value) for value in choice_sizes})
            )
            top_l_actions = {}
            action_modes = [0] * len(valid_indices)
            action_size_ids = [0] * len(valid_indices)
            for compact_index, flat_index in enumerate(valid_indices):
                mode = int(flat_index) // num_choices
                choice_index = int(action_choices[compact_index])
                if (
                    mode < 0
                    or mode >= num_modes
                    or choice_index < 0
                    or choice_index >= num_choices
                ):
                    raise ValueError(
                        "Invalid paper_factorized action metadata"
                    )
                size = int(choice_sizes[choice_index])
                ranks = tuple(int(v) for v in choice_ranks[choice_index])
                size_id = size_options.index(size)
                action_modes[compact_index] = mode
                action_size_ids[compact_index] = size_id
                if mode < 2 and ranks == tuple(range(size)):
                    top_l_actions[(mode, size_id)] = compact_index
            missing = [
                (mode, size)
                for mode in range(2)
                for size in range(len(size_options))
                if (mode, size) not in top_l_actions
            ]
            if missing:
                raise ValueError(
                    "Missing top-L action for paper branches: "
                    f"{missing}"
                )
            nc_actions = [
                index
                for index, mode in enumerate(action_modes)
                if mode == 2
            ]
            if len(nc_actions) != 1:
                raise ValueError(
                    "paper_factorized expects exactly one NC action"
                )
            action_size_lut = [
                [
                    top_l_actions.get((mode, size), nc_actions[0])
                    for size in range(len(size_options))
                ]
                for mode in range(num_modes)
            ]
            self.act = PaperFactorizedACTLayer(
                action_space,
                self.hidden_size,
                num_modes=num_modes,
                size_options=size_options,
                action_size_indices=action_size_lut,
                action_modes=action_modes,
                action_size_indices_by_action=action_size_ids,
                use_orthogonal=self._use_orthogonal,
                gain=self._gain,
            )
            self._paper_factorized = True
        if (
            not self._paper_factorized
            and self.joint_head_type in {
                "subset",
                "subset_conditional",
                "relational_subset",
                "mlp_factorized",
            }
            and nvec is not None
            and len(nvec) == 1
        ):
            num_modes = int(getattr(args, "joint_num_modes", 3))
            num_choices = int(
                getattr(args, "joint_num_cluster_choices", 1)
            )
            ranks = getattr(args, "joint_cluster_choice_ranks", None)
            choice_sizes = getattr(
                args, "joint_cluster_choice_sizes", None
            )
            n_ap_tokens = int(getattr(args, "joint_num_ap_tokens", 0))
            valid_indices = getattr(
                args, "joint_valid_action_indices", None
            )
            if (
                ranks is None
                or choice_sizes is None
                or n_ap_tokens <= 0
            ):
                raise ValueError(
                    "subset action head requires cluster-rank metadata "
                    "and the number of AP tokens"
                )
            if valid_indices is None:
                valid_indices = list(range(int(nvec[0])))
            if len(valid_indices) == int(nvec[0]):
                self._score_ap_tokens = (
                    self.joint_head_type != "mlp_factorized"
                )
                if self._score_ap_tokens:
                    token_input_dim = int(
                        getattr(args, "tf_d_model", 64)
                        if self.backbone_name == "transformer"
                        else getattr(args, "actor_token_dim", 1)
                    )
                    init_method = [
                        nn.init.xavier_uniform_, nn.init.orthogonal_
                    ][self._use_orthogonal]
                    token_proj = init(
                        nn.Linear(token_input_dim, self.hidden_size),
                        init_method,
                        lambda x: nn.init.constant_(x, 0),
                    )
                    self.subset_token_encoder = nn.Sequential(
                        nn.LayerNorm(token_input_dim),
                        token_proj,
                        nn.SiLU(),
                    )
                else:
                    self.subset_token_encoder = None
                if (
                    self.joint_head_type == "relational_subset"
                    and self.backbone_name != "transformer"
                ):
                    raise ValueError(
                        "relational_subset requires the Transformer actor"
                    )
                if (
                    self.joint_head_type == "mlp_factorized"
                    and self.backbone_name != "mlp"
                ):
                    raise ValueError(
                        "mlp_factorized requires the MLP actor"
                    )
                head_cls = (
                    RelationalSubsetACTLayer
                    if self.joint_head_type == "relational_subset"
                    else StructuredSubsetACTLayer
                )
                head_kwargs = dict(
                    num_modes=num_modes,
                    num_choices=num_choices,
                    cluster_choice_ranks=ranks,
                    cluster_choice_sizes=choice_sizes,
                    num_ap_tokens=n_ap_tokens,
                    use_orthogonal=self._use_orthogonal,
                    gain=self._gain,
                    valid_action_indices=valid_indices,
                    conditional_modes=(
                        self.joint_head_type
                        in {
                            "subset_conditional",
                            "relational_subset",
                            "mlp_factorized",
                        }
                    ),
                    actor_temporal_context=str(
                        getattr(
                            args,
                            "actor_temporal_context",
                            "history_mean",
                        )
                    ),
                    tokens_per_frame=1 + n_ap_tokens,
                    score_from_tokens=self._score_ap_tokens,
                )
                if head_cls is RelationalSubsetACTLayer:
                    head_kwargs["relation_dim"] = int(
                        getattr(args, "tf_relation_dim", 32)
                    )
                self.act = head_cls(
                    action_space,
                    self.hidden_size,
                    **head_kwargs,
                )
                self._structured_subset = True
        if (
            not self._paper_factorized
            and self.joint_head_type == "gated"
            and nvec is not None
            and len(nvec) == 1
        ) and not self._structured_subset:
            num_modes = int(getattr(args, "joint_num_modes", 3))
            num_sizes = int(getattr(
                args, "joint_num_sizes", max(1, int(nvec[0]) // num_modes)
            ))
            valid_indices = getattr(
                args, "joint_valid_action_indices", None
            )
            if valid_indices is None:
                valid_indices = list(range(int(nvec[0])))
            if len(valid_indices) == int(nvec[0]):
                self._gated_joint = True
                self.act = GatedJointACTLayer(
                    action_space,
                    self.hidden_size,
                    num_modes=num_modes,
                    num_sizes=num_sizes,
                    use_orthogonal=self._use_orthogonal,
                    gain=self._gain,
                    token_repr_dim=(
                        int(getattr(args, "tf_d_model", 64))
                        if self.backbone_name == "transformer"
                        else None
                    ),
                    gate_hidden=int(
                        getattr(args, "joint_gate_hidden", 0)
                        or self.hidden_size
                    ),
                    valid_action_indices=valid_indices,
                )
        if (
            not self._paper_factorized
            and self.joint_head_type == "flat"
            and nvec is not None
            and len(nvec) == 1
        ):
            self.act = FlatJointACTLayer(
                action_space,
                self.hidden_size,
                self._use_orthogonal,
                self._gain,
            )
        elif (
            not self._gated_joint
            and not self._structured_subset
            and not self._paper_factorized
        ):
            self.act = ACTLayer(
                action_space,
                self.hidden_size,
                self._use_orthogonal,
                self._gain,
            )

        self.to(device)

    def _encode_actor(self, obs):
        if self._structured_subset:
            if self.backbone_name == "transformer":
                pooled, raw_tokens = self.base.forward_with_tokens(obs)
            else:
                pooled = self.base(obs)
                if not self._score_ap_tokens:
                    return pooled, None
                token_dim = int(
                    getattr(self, "_actor_token_dim", 1)
                )
                raw_tokens = obs.view(
                    obs.size(0), -1, token_dim
                )
            return pooled, self.subset_token_encoder(raw_tokens)
        if self._gated_joint and self.backbone_name == "transformer":
            return self.base.forward_with_tokens(obs)
        return self.base(obs), None

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features, actor_tokens = self._encode_actor(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if (
            self._gated_joint
            or self._structured_subset
            or self._paper_factorized
        ):
            actions, action_log_probs = self.act(
                actor_features,
                available_actions,
                deterministic,
                tokens=actor_tokens,
            )
        else:
            actions, action_log_probs = self.act(
                actor_features,
                available_actions,
                deterministic,
            )

        return actions, action_log_probs, rnn_states

    def load_state_dict(self, state_dict, strict=True, assign=False):
        state_dict = _remap_state(state_dict)
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:
            return super().load_state_dict(state_dict, strict=strict)

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features, actor_tokens = self._encode_actor(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if (
            self._gated_joint
            or self._structured_subset
            or self._paper_factorized
        ):
            action_log_probs, dist_entropy = self.act.evaluate_actions(
                actor_features,
                action,
                available_actions,
                active_masks=(
                    active_masks
                    if self._use_policy_active_masks
                    else None
                ),
                tokens=actor_tokens,
            )
        else:
            action_log_probs, dist_entropy = self.act.evaluate_actions(
                actor_features,
                action,
                available_actions,
                active_masks=(
                    active_masks
                    if self._use_policy_active_masks
                    else None
                ),
            )

        return action_log_probs, dist_entropy

    def num_params(self):
        return count_parameters(self)


class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.backbone_name = str(getattr(args, "critic_backbone", "mlp")).lower()
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        hidden_override = int(getattr(args, "critic_hidden_size", 0) or 0)
        self.hidden_size = hidden_override if hidden_override > 0 else self.hidden_size
        self.base = _make_base(
            args, cent_obs_shape, "critic",
            hidden_override=(hidden_override if hidden_override > 0 else None),
        )

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states

    def load_state_dict(self, state_dict, strict=True, assign=False):
        state_dict = _remap_state(state_dict)
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:
            return super().load_state_dict(state_dict, strict=strict)

    def num_params(self):
        return count_parameters(self)
