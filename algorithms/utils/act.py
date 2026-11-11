import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .distributions import Bernoulli, Categorical, DiagGaussian, FixedCategorical
from .util import init

class PaperFactorizedACTLayer(nn.Module):

    def __init__(self, action_space, inputs_dim, num_modes, size_options, action_size_indices, action_modes, action_size_indices_by_action, use_orthogonal=True, gain=0.01):
        super().__init__()
        nvec = getattr(action_space, 'nvec', None)
        if nvec is None or len(nvec) != 1:
            raise ValueError('PaperFactorizedACTLayer expects one compact joint action')
        self.num_modes = int(num_modes)
        self.size_options = tuple((int(value) for value in size_options))
        if self.num_modes != 3 or not self.size_options:
            raise ValueError('PaperFactorizedACTLayer expects three modes and at least one cooperative cluster size')
        self.num_sizes = len(self.size_options)
        self.num_actions = int(nvec[0])
        size_map = torch.as_tensor(action_size_indices, dtype=torch.long)
        if size_map.shape != (self.num_modes, self.num_sizes):
            raise ValueError(f'action_size_indices must have shape ({self.num_modes}, {self.num_sizes})')
        self.register_buffer('action_size_lut', size_map, persistent=False)
        mode_by_action = torch.as_tensor(action_modes, dtype=torch.long)
        size_by_action = torch.as_tensor(action_size_indices_by_action, dtype=torch.long)
        if mode_by_action.numel() != self.num_actions or size_by_action.numel() != self.num_actions:
            raise ValueError('Per-action branch metadata must cover every compact action')
        self.register_buffer('mode_by_action', mode_by_action, persistent=False)
        self.register_buffer('size_by_action', size_by_action, persistent=False)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(module):
            return init(module, init_method, lambda value: nn.init.constant_(value, 0), gain)
        self.mode_head = init_(nn.Linear(inputs_dim, self.num_modes))
        self.size_head = init_(nn.Linear(inputs_dim, self.num_sizes))

    def branch_distributions(self, x):
        return (FixedCategorical(logits=self.mode_head(x)), FixedCategorical(logits=self.size_head(x)))

    def _map_branches(self, modes, sizes):
        modes = modes.long().view(-1)
        sizes = sizes.long().view(-1)
        safe_modes = modes.clamp(0, self.num_modes - 1)
        safe_sizes = sizes.clamp(0, self.num_sizes - 1)
        return self.action_size_lut[safe_modes, safe_sizes].view(-1, 1)

    def forward(self, x, available_actions=None, deterministic=False, tokens=None):
        if available_actions is not None:
            raise ValueError('PaperFactorizedACTLayer uses branch-level action validity')
        (mode_dist, size_dist) = self.branch_distributions(x)
        mode = mode_dist.mode() if deterministic else mode_dist.sample()
        size = size_dist.mode() if deterministic else size_dist.sample()
        action = self._map_branches(mode, size)
        log_prob = mode_dist.log_probs(mode) + size_dist.log_probs(size)
        return (action, log_prob)

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None, tokens=None):
        if available_actions is not None:
            raise ValueError('PaperFactorizedACTLayer uses branch-level action validity')
        (mode_dist, size_dist) = self.branch_distributions(x)
        action_index = action.long().view(-1)
        if torch.any(action_index < 0) or torch.any(action_index >= self.num_actions):
            raise ValueError('Compact joint action is out of range')
        mode = self.mode_by_action[action_index].view(-1, 1)
        size = self.size_by_action[action_index].view(-1, 1)
        log_prob = mode_dist.log_probs(mode) + size_dist.log_probs(size)
        entropy = 0.5 * (mode_dist.entropy() + size_dist.entropy())
        if active_masks is not None:
            entropy = (entropy * active_masks.squeeze(-1)).sum() / active_masks.sum()
        else:
            entropy = entropy.mean()
        return (log_prob, entropy)

class GatedJointACTLayer(nn.Module):

    def __init__(self, action_space, inputs_dim, num_modes, num_sizes, use_orthogonal=True, gain=0.01, token_repr_dim=None, gate_hidden=None, valid_action_indices=None):
        super().__init__()
        nvec = getattr(action_space, 'nvec', None)
        if nvec is None or len(nvec) != 1:
            raise ValueError('GatedJointACTLayer expects one joint action variable')
        self.num_modes = int(num_modes)
        self.num_sizes = int(num_sizes)
        self.grid_actions = self.num_modes * self.num_sizes
        if valid_action_indices is None:
            valid_action_indices = list(range(self.grid_actions))
        valid_indices = [int(value) for value in valid_action_indices]
        if not valid_indices or len(set(valid_indices)) != len(valid_indices) or min(valid_indices) < 0 or (max(valid_indices) >= self.grid_actions):
            raise ValueError('Invalid joint action validity mask')
        self.register_buffer('valid_action_indices', torch.as_tensor(valid_indices, dtype=torch.long), persistent=False)
        valid_grid_mask = torch.zeros(self.grid_actions, dtype=torch.bool)
        valid_grid_mask[self.valid_action_indices] = True
        self.register_buffer('valid_grid_mask', valid_grid_mask, persistent=False)
        self.num_actions = len(valid_indices)
        if int(nvec[0]) != self.num_actions:
            raise ValueError(f'Joint action space expects {self.num_actions} actions, got {int(nvec[0])}')
        self.uses_tokens = token_repr_dim is not None
        self.branch_dim = int(token_repr_dim if self.uses_tokens else inputs_dim)
        gate_dim = int(gate_hidden or self.branch_dim)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(module, output_gain=1.0):
            return init(module, init_method, lambda x: nn.init.constant_(x, 0), output_gain)
        if self.uses_tokens:
            self.mode_query = nn.Parameter(torch.randn(self.branch_dim) * 0.02)
            self.size_query = nn.Parameter(torch.randn(self.branch_dim) * 0.02)
            self.mode_pool_residual = init_(nn.Linear(inputs_dim, self.branch_dim))
            self.size_pool_residual = init_(nn.Linear(inputs_dim, self.branch_dim))
        else:
            self.mode_encoder = init_(nn.Linear(inputs_dim, self.branch_dim))
            self.size_encoder = init_(nn.Linear(inputs_dim, self.branch_dim))
        self.gate_net = nn.Sequential(init_(nn.Linear(2 * self.branch_dim, gate_dim)), nn.SiLU(), init_(nn.Linear(gate_dim, 2 * self.branch_dim)))
        nn.init.constant_(self.gate_net[-1].bias, 0.0)
        self.fusion = nn.Sequential(nn.LayerNorm(2 * self.branch_dim), init_(nn.Linear(2 * self.branch_dim, self.branch_dim)), nn.SiLU())
        self.mode_head = init_(nn.Linear(self.branch_dim, self.num_modes), output_gain=gain)
        self.size_head = init_(nn.Linear(self.branch_dim, self.num_sizes), output_gain=gain)
        self.interaction_head = init_(nn.Linear(self.branch_dim, self.grid_actions), output_gain=gain)
        self.last_gate = None

    @staticmethod
    def _token_attention(tokens: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(tokens, query) / math.sqrt(float(tokens.size(-1)))
        weights = torch.softmax(scores, dim=1)
        return torch.sum(tokens * weights.unsqueeze(-1), dim=1)

    def _branch_representations(self, x, tokens=None):
        if self.uses_tokens:
            if tokens is None:
                raise ValueError('GatedJointACTLayer requires Transformer tokens')
            mode_context = self._token_attention(tokens, self.mode_query) + self.mode_pool_residual(x)
            size_context = self._token_attention(tokens, self.size_query) + self.size_pool_residual(x)
            return (mode_context, size_context)
        return (self.mode_encoder(x), self.size_encoder(x))

    def joint_logits(self, x, tokens=None, available_actions=None):
        (mode_repr, size_repr) = self._branch_representations(x, tokens)
        gate = torch.sigmoid(self.gate_net(torch.cat((mode_repr, size_repr), dim=-1)))
        (gate_mode, gate_size) = torch.chunk(gate, 2, dim=-1)
        self.last_gate = gate.detach()
        mode_enhanced = mode_repr * gate_mode + size_repr * (1.0 - gate_mode)
        size_enhanced = size_repr * gate_size + mode_repr * (1.0 - gate_size)
        fused = self.fusion(torch.cat((mode_enhanced, size_enhanced), dim=-1))
        mode_logits = self.mode_head(mode_enhanced + fused)
        size_logits = self.size_head(size_enhanced + fused)
        interaction = self.interaction_head(fused).view(-1, self.num_modes, self.num_sizes)
        joint = (mode_logits[:, :, None] + size_logits[:, None, :] + interaction).view(-1, self.grid_actions)
        joint = joint.masked_fill(~self.valid_grid_mask.unsqueeze(0), -10000000000.0)
        joint = joint[:, self.valid_action_indices]
        if available_actions is not None:
            joint = joint.masked_fill(available_actions == 0, -10000000000.0)
        return joint

    def forward(self, x, available_actions=None, deterministic=False, tokens=None):
        distribution = FixedCategorical(logits=self.joint_logits(x, tokens=tokens, available_actions=available_actions))
        action = distribution.mode() if deterministic else distribution.sample()
        return (action, distribution.log_probs(action))

    def get_probs(self, x, available_actions=None, tokens=None):
        return F.softmax(self.joint_logits(x, tokens=tokens, available_actions=available_actions), dim=-1)

    def branch_probabilities(self, x, available_actions=None, tokens=None):
        probs = self.get_probs(x, available_actions=available_actions, tokens=tokens)
        full_probs = probs.new_zeros((probs.size(0), self.grid_actions))
        full_probs[:, self.valid_action_indices] = probs
        probs = full_probs.view(-1, self.num_modes, self.num_sizes)
        return (probs.sum(dim=2), probs.sum(dim=1), probs)

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None, tokens=None):
        distribution = FixedCategorical(logits=self.joint_logits(x, tokens=tokens, available_actions=available_actions))
        action_log_probs = distribution.log_probs(action)
        entropy = distribution.entropy()
        if active_masks is not None:
            entropy = (entropy * active_masks.squeeze(-1)).sum() / active_masks.sum()
        else:
            entropy = entropy.mean()
        return (action_log_probs, entropy)

class StructuredSubsetACTLayer(nn.Module):

    def __init__(self, action_space, inputs_dim, num_modes, num_choices, cluster_choice_ranks, cluster_choice_sizes, num_ap_tokens, use_orthogonal=True, gain=0.01, valid_action_indices=None, conditional_modes=False, actor_temporal_context='history_mean', tokens_per_frame=1, score_from_tokens=True):
        super().__init__()
        nvec = getattr(action_space, 'nvec', None)
        if nvec is None or len(nvec) != 1:
            raise ValueError('StructuredSubsetACTLayer expects one joint action variable')
        self.num_modes = int(num_modes)
        self.cooperative_modes = 2
        self.conditional_modes = bool(conditional_modes)
        self.actor_temporal_context = str(actor_temporal_context)
        if self.actor_temporal_context not in {'history_mean', 'current_frame', 'current_frame_balanced'}:
            raise ValueError('actor_temporal_context must be history_mean, current_frame, or current_frame_balanced')
        self.tokens_per_frame = int(tokens_per_frame)
        self.score_from_tokens = bool(score_from_tokens)
        if self.tokens_per_frame < 1:
            raise ValueError('tokens_per_frame must be positive')
        self.num_choices = int(num_choices)
        self.num_ap_tokens = int(num_ap_tokens)
        self.grid_actions = self.num_modes * self.num_choices
        if valid_action_indices is None:
            valid_action_indices = list(range(self.grid_actions))
        valid_indices = [int(v) for v in valid_action_indices]
        if not valid_indices or len(set(valid_indices)) != len(valid_indices) or min(valid_indices) < 0 or (max(valid_indices) >= self.grid_actions):
            raise ValueError('Invalid structured subset action mask')
        self.register_buffer('valid_action_indices', torch.as_tensor(valid_indices, dtype=torch.long), persistent=False)
        valid_mask = torch.zeros(self.grid_actions, dtype=torch.bool)
        valid_mask[self.valid_action_indices] = True
        self.register_buffer('valid_grid_mask', valid_mask, persistent=False)
        self.num_actions = len(valid_indices)
        if int(nvec[0]) != self.num_actions:
            raise ValueError(f'Joint action space expects {self.num_actions} actions, got {int(nvec[0])}')
        if len(cluster_choice_ranks) != self.num_choices:
            raise ValueError('cluster_choice_ranks must match the number of choices')
        if len(cluster_choice_sizes) != self.num_choices:
            raise ValueError('cluster_choice_sizes must match the number of choices')
        unique_sizes = list(dict.fromkeys((int(size) for size in cluster_choice_sizes)))
        size_to_index = {size: index for (index, size) in enumerate(unique_sizes)}
        self.num_size_options = len(unique_sizes)
        membership = torch.zeros(self.num_choices, self.num_ap_tokens, dtype=torch.float32)
        for (index, ranks) in enumerate(cluster_choice_ranks):
            for rank in ranks:
                rank = int(rank)
                if not 0 <= rank < self.num_ap_tokens:
                    raise ValueError(f'AP rank {rank} is outside the candidate pool')
                membership[index, rank] = 1.0
        self.register_buffer('choice_membership', membership, persistent=False)
        self.register_buffer('choice_member_count', membership.sum(dim=1).clamp_min(1.0), persistent=False)
        self.register_buffer('choice_size_index', torch.as_tensor([size_to_index[int(size)] for size in cluster_choice_sizes], dtype=torch.long), persistent=False)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(module, output_gain=1.0):
            return init(module, init_method, lambda x: nn.init.constant_(x, 0), output_gain)
        self.mode_head = init_(nn.Linear(inputs_dim, self.num_modes), output_gain=gain)
        score_dim = self.cooperative_modes if self.conditional_modes else 1
        ap_output_dim = score_dim * (1 if self.score_from_tokens else self.num_ap_tokens)
        self.ap_score_head = init_(nn.Linear(inputs_dim, ap_output_dim), output_gain=gain)
        self.size_head = init_(nn.Linear(inputs_dim, self.num_size_options * (self.cooperative_modes if self.conditional_modes else 1)), output_gain=gain)
        self.choice_bias = nn.Parameter(torch.zeros(self.cooperative_modes if self.conditional_modes else 1, self.num_choices))
        self.last_mode_probs = None
        self.last_choice_probs = None

    def _choice_values(self, x, tokens, context=None):
        if self.score_from_tokens:
            if tokens is None:
                raise ValueError('StructuredSubsetACTLayer requires candidate AP tokens')
            if tokens.size(1) < 1 + self.num_ap_tokens:
                raise ValueError('Not enough AP tokens for structured subset actions')
            candidate_tokens = tokens[:, -self.num_ap_tokens:, :]
            ap_scores = self.ap_score_head(candidate_tokens)
        else:
            score_dim = self.cooperative_modes if self.conditional_modes else 1
            ap_scores = self.ap_score_head(x).view(x.size(0), self.num_ap_tokens, score_dim)
        if context is None:
            context = self._context_features(x, tokens)
        if not self.conditional_modes:
            ap_scores = ap_scores.squeeze(-1)
            ap_sums = torch.matmul(ap_scores, self.choice_membership.t()).unsqueeze(1)
            size_logits = self.size_head(context).unsqueeze(1)
        else:
            ap_sums = torch.einsum('bnk,cn->bkc', ap_scores, self.choice_membership)
            size_logits = self.size_head(context).view(context.size(0), self.cooperative_modes, self.num_size_options)
        choice_values = ap_sums / self.choice_member_count.view(1, 1, -1) + size_logits[:, :, self.choice_size_index] + self.choice_bias.unsqueeze(0)
        if not self.conditional_modes:
            choice_values = choice_values.expand(-1, self.cooperative_modes, -1)
        return choice_values

    def _context_features(self, x, tokens):
        if self.actor_temporal_context == 'history_mean':
            return x
        if tokens is None:
            raise ValueError('current_frame temporal context requires token outputs')
        if tokens.size(1) < self.tokens_per_frame:
            raise ValueError('Not enough tokens for the current observation frame')
        current_frame = tokens[:, -self.tokens_per_frame:, :]
        if self.actor_temporal_context == 'current_frame':
            return current_frame.mean(dim=1)
        ue_context = current_frame[:, 0, :]
        ap_context = current_frame[:, 1:, :].mean(dim=1)
        return 0.5 * (ue_context + ap_context)

    def joint_logits(self, x, tokens=None, available_actions=None):
        context = self._context_features(x, tokens)
        choice_values = self._choice_values(x, tokens, context=context)
        mode_log_probs = F.log_softmax(self.mode_head(context), dim=-1)
        choice_log_probs = F.log_softmax(choice_values, dim=-1)
        self.last_mode_probs = mode_log_probs.detach().exp()
        self.last_choice_probs = choice_log_probs.detach().exp()
        joint = torch.zeros(x.size(0), self.num_modes, self.num_choices, device=x.device, dtype=x.dtype)
        joint[:, 0, :] = mode_log_probs[:, 0:1] + choice_log_probs[:, 0, :]
        joint[:, 1, :] = mode_log_probs[:, 1:2] + choice_log_probs[:, 1, :]
        joint[:, 2, 0] = mode_log_probs[:, 2]
        joint = joint.view(-1, self.grid_actions)
        joint = joint.masked_fill(~self.valid_grid_mask.unsqueeze(0), -10000000000.0)
        joint = joint[:, self.valid_action_indices]
        if available_actions is not None:
            joint = joint.masked_fill(available_actions == 0, -10000000000.0)
        return joint

    def forward(self, x, available_actions=None, deterministic=False, tokens=None):
        logits = self.joint_logits(x, tokens=tokens, available_actions=available_actions)
        distribution = FixedCategorical(logits=logits)
        if deterministic:
            if self.score_from_tokens and tokens is None:
                raise ValueError('Deterministic subset actions require AP tokens')
            context = self._context_features(x, tokens)
            choice_values = self._choice_values(x, tokens, context=context)
            mode = torch.argmax(self.mode_head(context), dim=-1)
            choice_logits = torch.where(mode[:, None] == 0, choice_values[:, 0, :], choice_values[:, 1, :])
            choice = torch.argmax(choice_logits, dim=-1)
            action = torch.where(mode == 2, torch.full_like(mode, 2 * self.num_choices), mode * self.num_choices + choice)
            valid_index = torch.searchsorted(self.valid_action_indices, action)
            action = valid_index.long().unsqueeze(-1)
        else:
            action = distribution.sample()
        return (action, distribution.log_probs(action))

    def get_probs(self, x, available_actions=None, tokens=None):
        return F.softmax(self.joint_logits(x, tokens=tokens, available_actions=available_actions), dim=-1)

    def branch_probabilities(self, x, available_actions=None, tokens=None):
        probs = self.get_probs(x, available_actions=available_actions, tokens=tokens)
        full_probs = probs.new_zeros((probs.size(0), self.grid_actions))
        full_probs[:, self.valid_action_indices] = probs
        joint = full_probs.view(-1, self.num_modes, self.num_choices)
        return (joint.sum(dim=2), joint[:, :2, :].sum(dim=1), joint)

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None, tokens=None):
        distribution = FixedCategorical(logits=self.joint_logits(x, tokens=tokens, available_actions=available_actions))
        action_log_probs = distribution.log_probs(action)
        entropy = distribution.entropy()
        if active_masks is not None:
            entropy = (entropy * active_masks.squeeze(-1)).sum() / active_masks.sum()
        else:
            entropy = entropy.mean()
        return (action_log_probs, entropy)

class RelationalSubsetACTLayer(StructuredSubsetACTLayer):

    def __init__(self, *args, relation_dim=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.relation_dim = max(1, int(relation_dim))

        def projection(input_dim, output_dim):
            layer = nn.Linear(input_dim, output_dim)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
            return layer
        context_dim = int(self.mode_head.in_features)
        self.context_query = projection(context_dim, self.relation_dim)
        self.context_key = projection(context_dim, self.relation_dim)
        self.context_value = projection(context_dim, self.relation_dim)
        self.context_out = projection(self.relation_dim, context_dim)
        self.ap_query = projection(context_dim, self.relation_dim)
        self.ap_key = projection(context_dim, self.relation_dim)
        self.ap_value = projection(context_dim, self.relation_dim)
        self.ap_out = projection(self.relation_dim, context_dim)
        self.ap_ue_context = projection(context_dim, context_dim)

    @staticmethod
    def _scaled_attention(query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(float(query.size(-1)))
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, value)

    def _current_frame(self, tokens):
        if tokens is None:
            raise ValueError('RelationalSubsetACTLayer requires Transformer tokens')
        if tokens.size(1) < self.tokens_per_frame:
            raise ValueError('Not enough tokens for the relational AP head')
        return tokens[:, -self.tokens_per_frame:, :]

    def _context_features(self, x, tokens):
        current_frame = self._current_frame(tokens)
        ue_token = current_frame[:, 0, :]
        query = self.context_query(ue_token).unsqueeze(1)
        key = self.context_key(tokens)
        value = self.context_value(tokens)
        attended = self._scaled_attention(query, key, value)
        return x + self.context_out(attended.squeeze(1))

    def _choice_values(self, x, tokens, context=None):
        current_frame = self._current_frame(tokens)
        candidate_tokens = current_frame[:, -self.num_ap_tokens:, :]
        if candidate_tokens.size(1) != self.num_ap_tokens:
            raise ValueError('Relational AP head expects one current token per candidate AP')
        ue_token = current_frame[:, 0:1, :]
        query = self.ap_query(candidate_tokens)
        key = self.ap_key(candidate_tokens)
        value = self.ap_value(candidate_tokens)
        relation = self._scaled_attention(query, key, value)
        relational_tokens = candidate_tokens + self.ap_out(relation) + self.ap_ue_context(ue_token).expand_as(candidate_tokens)
        ap_scores = self.ap_score_head(relational_tokens)
        if context is None:
            context = self._context_features(x, tokens)
        ap_sums = torch.einsum('bnk,cn->bkc', ap_scores, self.choice_membership)
        size_logits = self.size_head(context).view(context.size(0), self.cooperative_modes, self.num_size_options)
        return ap_sums / self.choice_member_count.view(1, 1, -1) + size_logits[:, :, self.choice_size_index] + self.choice_bias.unsqueeze(0)

class FlatJointACTLayer(nn.Module):

    def __init__(self, action_space, inputs_dim, use_orthogonal=True, gain=0.01):
        super().__init__()
        nvec = getattr(action_space, 'nvec', None)
        if nvec is None or len(nvec) != 1:
            raise ValueError('FlatJointACTLayer expects one discrete joint action')
        self.action_out = Categorical(inputs_dim, int(nvec[0]), use_orthogonal, gain)

    def forward(self, x, available_actions=None, deterministic=False, tokens=None):
        distribution = self.action_out(x, available_actions)
        action = distribution.mode() if deterministic else distribution.sample()
        return (action, distribution.log_probs(action))

    def get_probs(self, x, available_actions=None, tokens=None):
        return self.action_out(x, available_actions).probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None, tokens=None):
        distribution = self.action_out(x, available_actions)
        action_log_probs = distribution.log_probs(action)
        entropy = distribution.entropy()
        if active_masks is not None:
            entropy = (entropy * active_masks.squeeze(-1)).sum() / active_masks.sum()
        else:
            entropy = entropy.mean()
        return (action_log_probs, entropy)

class ACTLayer(nn.Module):

    def __init__(self, action_space, inputs_dim, use_orthogonal, gain):
        super(ACTLayer, self).__init__()
        self.mixed_action = False
        self.multi_discrete = False
        self.continuous_action = False
        if action_space.__class__.__name__ == 'Discrete':
            action_dim = action_space.n
            self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == 'Box':
            self.continuous_action = True
            action_dim = action_space.shape[0]
            self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == 'MultiBinary':
            action_dim = action_space.shape[0]
            self.action_out = Bernoulli(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == 'MultiDiscrete':
            self.multi_discrete = True
            action_dims = action_space.nvec
            self.action_outs = []
            for action_dim in action_dims:
                self.action_outs.append(Categorical(inputs_dim, action_dim, use_orthogonal, gain))
            self.action_outs = nn.ModuleList(self.action_outs)
        else:
            self.mixed_action = True
            continous_dim = action_space[0].shape[0]
            discrete_dim = action_space[1].n
            self.action_outs = nn.ModuleList([DiagGaussian(inputs_dim, continous_dim, use_orthogonal, gain), Categorical(inputs_dim, discrete_dim, use_orthogonal, gain)])

    def forward(self, x, available_actions=None, deterministic=False):
        if self.mixed_action:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                actions.append(action.float())
                action_log_probs.append(action_log_prob)
            actions = torch.cat(actions, -1)
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
        elif self.multi_discrete:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)
            actions = torch.cat(actions, -1)
            action_log_probs = torch.cat(action_log_probs, -1)
        elif self.continuous_action:
            action_logit = self.action_out(x)
            actions = action_logit.mode() if deterministic else action_logit.sample()
            action_log_probs = action_logit.log_probs(actions)
        else:
            action_logits = self.action_out(x, available_actions)
            actions = action_logits.mode() if deterministic else action_logits.sample()
            action_log_probs = action_logits.log_probs(actions)
        return (actions, action_log_probs)

    def get_probs(self, x, available_actions=None):
        if self.mixed_action or self.multi_discrete:
            action_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action_prob = action_logit.probs
                action_probs.append(action_prob)
            action_probs = torch.cat(action_probs, -1)
        else:
            action_logits = self.action_out(x, available_actions)
            action_probs = action_logits.probs
        return action_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        if self.mixed_action:
            (a, b) = action.split((2, 1), -1)
            b = b.long()
            action = [a, b]
            action_log_probs = []
            dist_entropy = []
            for (action_out, act) in zip(self.action_outs, action):
                action_logit = action_out(x)
                action_log_probs.append(action_logit.log_probs(act))
                if active_masks is not None:
                    if len(action_logit.entropy().shape) == len(active_masks.shape):
                        dist_entropy.append((action_logit.entropy() * active_masks).sum() / active_masks.sum())
                    else:
                        dist_entropy.append((action_logit.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
            dist_entropy = dist_entropy[0] / 2.0 + dist_entropy[1] / 0.98
        elif self.multi_discrete:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            for (action_out, act) in zip(self.action_outs, action):
                action_logit = action_out(x)
                action_log_probs.append(action_logit.log_probs(act))
                if active_masks is not None:
                    dist_entropy.append((action_logit.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())
            action_log_probs = torch.cat(action_log_probs, -1)
            dist_entropy = torch.stack(dist_entropy).mean()
        elif self.continuous_action:
            action_log_probs = []
            dist_entropy = []
            action_logit = self.action_out(x)
            action_log_probs.append(action_logit.log_probs(action))
            if active_masks is not None:
                if len(action_logit.entropy().shape) == len(active_masks.shape):
                    dist_entropy.append((action_logit.entropy() * active_masks).sum() / active_masks.sum())
                else:
                    dist_entropy.append((action_logit.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum())
            else:
                dist_entropy.append(action_logit.entropy().mean())
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
            dist_entropy = dist_entropy[0]
        else:
            action_logits = self.action_out(x, available_actions)
            action_log_probs = action_logits.log_probs(action)
            if active_masks is not None:
                dist_entropy = (action_logits.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()
        return (action_log_probs, dist_entropy)
