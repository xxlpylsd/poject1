from __future__ import annotations
import sys
import unittest
from argparse import Namespace
from pathlib import Path
import numpy as np
import torch
from gym import spaces
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from algorithms.algorithm.r_actor_critic import R_Actor, R_Critic
from algorithms.utils.act import FlatJointACTLayer, RelationalSubsetACTLayer, StructuredSubsetACTLayer

def actor_args(backbone: str) -> Namespace:
    return Namespace(hidden_size=64, gain=0.01, use_orthogonal=True, use_policy_active_masks=True, use_naive_recurrent_policy=False, use_recurrent_policy=False, recurrent_N=1, actor_backbone=backbone, actor_token_dim=4, joint_action_head='gated', joint_num_modes=3, joint_num_sizes=3, joint_num_cluster_choices=3, joint_cluster_choice_ranks=[[0, 1], [0, 1, 2], [0, 1, 2, 3, 4]], joint_cluster_choice_sizes=[2, 3, 5], joint_num_ap_tokens=5, joint_num_actions=7, joint_valid_action_indices=[0, 1, 2, 3, 4, 5, 6], joint_gate_hidden=32, use_feature_normalization=True, use_ReLU=True, tf_d_model=64, tf_n_head=4, tf_n_layers=2, tf_ff_mult=4, tf_relation_dim=16, tf_dropout=0.0, tf_pool='mean', tf_use_cls=False, tf_use_pe=True, tf_max_tokens=256, tf_grad_checkpoint=False, layer_N=1, stacked_frames=1)

class GatedJointActorTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(7)
        self.obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([7])
        self.obs = torch.randn(5, 24)
        self.rnn = torch.zeros(5, 1, 1)
        self.masks = torch.ones(5, 1)

    def _check_actor(self, backbone: str):
        actor = R_Actor(actor_args(backbone), self.obs_space, self.action_space)
        actor.eval()
        with torch.no_grad():
            (actions, log_probs, _) = actor(self.obs, self.rnn, self.masks, deterministic=True)
        self.assertEqual(tuple(actions.shape), (5, 1))
        self.assertEqual(tuple(log_probs.shape), (5, 1))
        self.assertTrue(torch.all(actions >= 0))
        self.assertTrue(torch.all(actions < 7))
        self.assertIsNotNone(actor.act.last_gate)
        self.assertEqual(tuple(actor.act.last_gate.shape), (5, 128))
        self.assertTrue(torch.all(torch.isfinite(actor.act.last_gate)))
        self.assertTrue(torch.all(actor.act.last_gate >= 0.0))
        self.assertTrue(torch.all(actor.act.last_gate <= 1.0))
        if backbone == 'transformer':
            (pooled, tokens) = actor.base.forward_with_tokens(self.obs)
        else:
            (pooled, tokens) = (actor.base(self.obs), None)
        (mode_probs, size_probs, joint) = actor.act.branch_probabilities(pooled, tokens=tokens)
        self.assertEqual(tuple(mode_probs.shape), (5, 3))
        self.assertEqual(tuple(size_probs.shape), (5, 3))
        self.assertEqual(tuple(joint.shape), (5, 3, 3))
        self.assertTrue(torch.allclose(joint.sum(dim=(1, 2)), torch.ones(5), atol=1e-06))
        with torch.no_grad():
            (evaluated, entropy) = actor.evaluate_actions(self.obs, self.rnn, actions, self.masks, active_masks=torch.ones(5, 1))
        self.assertEqual(tuple(evaluated.shape), (5, 1))
        self.assertEqual(entropy.dim(), 0)

    def test_mlp_gated_joint_actor(self):
        self._check_actor('mlp')

    def test_transformer_gated_joint_actor(self):
        self._check_actor('transformer')

    def test_transformer_actor_can_use_cls_for_global_action_heads(self):
        args = actor_args('transformer')
        args.tf_pool = 'cls'
        args.tf_use_cls = True
        actor = R_Actor(args, self.obs_space, self.action_space)
        self.assertTrue(actor.base.backbone.use_cls)
        self.assertEqual(actor.base.backbone.pool, 'cls')
        actor.eval()
        with torch.no_grad():
            (pooled, tokens) = actor.base.forward_with_tokens(self.obs)
            mode_logits = actor.act.mode_head(pooled)
        self.assertEqual(tuple(pooled.shape), (5, args.hidden_size))
        self.assertEqual(tuple(tokens.shape), (5, 6, args.tf_d_model))
        self.assertEqual(tuple(mode_logits.shape), (5, 3))
        self.assertTrue(torch.all(torch.isfinite(mode_logits)))

    def test_transformer_critic_uses_one_token_per_ue(self):
        args = actor_args('mlp')
        args.critic_backbone = 'transformer'
        args.critic_token_dim = 144
        args.critic_hidden_size = 64
        args.use_popart = False
        cent_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4320,), dtype=np.float32)
        critic = R_Critic(args, cent_obs_space)
        critic.eval()
        self.assertEqual(critic.backbone_name, 'transformer')
        self.assertEqual(critic.base.backbone.n_tokens, 30)
        cent_obs = torch.randn(3, 4320)
        masks = torch.ones(3, 1)
        rnn = torch.zeros(3, 1, 1)
        with torch.no_grad():
            (values, _) = critic(cent_obs, rnn, masks)
        self.assertEqual(tuple(values.shape), (3, 1))
        self.assertTrue(torch.all(torch.isfinite(values)))

    def test_actor_supports_4_frame_6d_53_action_input(self):
        from itertools import combinations
        ranks = [list(choice) for size in (2, 3, 4, 5) for choice in combinations(range(5), size)]
        sizes = [size for size in (2, 3, 4, 5) for _choice in combinations(range(5), size)]
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(144,), dtype=np.float32)
        obs = torch.randn(3, 144)
        masks = torch.ones(3, 1)
        rnn = torch.zeros(3, 1, 1)
        for backbone in ('mlp', 'transformer'):
            args = actor_args(backbone)
            args.actor_token_dim = 6
            args.joint_action_head = 'subset_conditional'
            args.joint_num_cluster_choices = 26
            args.joint_cluster_choice_ranks = ranks
            args.joint_cluster_choice_sizes = sizes
            args.joint_num_ap_tokens = 5
            args.joint_num_actions = 53
            args.joint_valid_action_indices = list(range(26)) + list(range(26, 52)) + [52]
            actor = R_Actor(args, obs_space, spaces.MultiDiscrete([53]))
            actor.eval()
            with torch.no_grad():
                (actions, log_probs, _) = actor(obs, rnn, masks, deterministic=False)
                (pooled, tokens) = actor._encode_actor(obs)
                joint = actor.act.joint_logits(pooled, tokens=tokens)
            self.assertEqual(tuple(actions.shape), (3, 1))
            self.assertEqual(tuple(log_probs.shape), (3, 1))
            self.assertEqual(tuple(tokens.shape), (3, 24, 64))
            self.assertEqual(tuple(joint.shape), (3, 53))

    def test_actor_supports_neighbor_ue_and_ap_competition_tokens(self):
        from itertools import combinations
        ranks = [
            list(choice)
            for size in (2, 3, 4, 5)
            for choice in combinations(range(5), size)
        ]
        sizes = [
            size
            for size in (2, 3, 4, 5)
            for _choice in combinations(range(5), size)
        ]
        obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(324,),
            dtype=np.float32,
        )
        obs = torch.randn(3, 324)
        masks = torch.ones(3, 1)
        rnn = torch.zeros(3, 1, 1)
        args = actor_args('transformer')
        args.actor_token_dim = 9
        args.obs_neighbor_ues = 3
        args.joint_action_head = 'subset_conditional'
        args.joint_num_cluster_choices = 26
        args.joint_cluster_choice_ranks = ranks
        args.joint_cluster_choice_sizes = sizes
        args.joint_num_ap_tokens = 5
        args.joint_num_actions = 53
        args.joint_valid_action_indices = (
            list(range(26)) + list(range(26, 52)) + [52]
        )
        actor = R_Actor(
            args,
            obs_space,
            spaces.MultiDiscrete([53]),
        )
        actor.eval()
        with torch.no_grad():
            (actions, log_probs, _) = actor(
                obs,
                rnn,
                masks,
                deterministic=False,
            )
            (pooled, tokens) = actor._encode_actor(obs)
            joint = actor.act.joint_logits(pooled, tokens=tokens)
        self.assertEqual(actor.act.tokens_per_frame, 9)
        self.assertEqual(tuple(actions.shape), (3, 1))
        self.assertEqual(tuple(log_probs.shape), (3, 1))
        self.assertEqual(tuple(tokens.shape), (3, 36, 64))
        self.assertEqual(tuple(joint.shape), (3, 53))

    def test_joint_actor_respects_available_action_mask(self):
        actor = R_Actor(actor_args('transformer'), self.obs_space, self.action_space)
        actor.eval()
        available = np.eye(7, dtype=np.float32)[3][None, :]
        available = np.repeat(available, 5, axis=0)
        with torch.no_grad():
            (actions, _, _) = actor(self.obs, self.rnn, self.masks, available_actions=available, deterministic=True)
        self.assertTrue(torch.all(actions == 3))

    def test_transformer_gate_depends_on_token_observations(self):
        actor = R_Actor(actor_args('transformer'), self.obs_space, self.action_space)
        actor.eval()
        with torch.no_grad():
            (pooled_a, tokens_a) = actor.base.forward_with_tokens(self.obs)
            actor.act.joint_logits(pooled_a, tokens=tokens_a)
            gate_a = actor.act.last_gate.clone()
            (pooled_b, tokens_b) = actor.base.forward_with_tokens(torch.zeros_like(self.obs) + 2.0)
            actor.act.joint_logits(pooled_b, tokens=tokens_b)
            gate_b = actor.act.last_gate.clone()
        self.assertFalse(torch.allclose(gate_a, gate_b))

    def test_gated_actor_supports_learned_cluster_subsets(self):
        args = actor_args('transformer')
        args.joint_num_sizes = 21
        args.joint_num_actions = 43
        valid = list(range(21)) + list(range(21, 42)) + [42]
        args.joint_valid_action_indices = valid
        action_space = spaces.MultiDiscrete([43])
        actor = R_Actor(args, self.obs_space, action_space)
        actor.eval()
        with torch.no_grad():
            (actions, log_probs, _) = actor(self.obs, self.rnn, self.masks, deterministic=True)
            (pooled, tokens) = actor.base.forward_with_tokens(self.obs)
            (mode_probs, choice_probs, joint) = actor.act.branch_probabilities(pooled, tokens=tokens)
        self.assertEqual(tuple(actions.shape), (5, 1))
        self.assertEqual(tuple(log_probs.shape), (5, 1))
        self.assertTrue(torch.all(actions < 43))
        self.assertEqual(tuple(mode_probs.shape), (5, 3))
        self.assertEqual(tuple(choice_probs.shape), (5, 21))
        self.assertEqual(tuple(joint.shape), (5, 3, 21))
        self.assertTrue(torch.allclose(joint.sum(dim=(1, 2)), torch.ones(5), atol=1e-06))

    def test_structured_subset_actor_uses_ap_token_scores(self):
        from itertools import combinations
        ranks = [list(choice) for size in (2, 3, 5) for choice in combinations(range(5), size)]
        sizes = [size for size in (2, 3, 5) for _choice in combinations(range(5), size)]
        for backbone in ('mlp', 'transformer'):
            args = actor_args(backbone)
            args.joint_action_head = 'subset'
            args.joint_num_cluster_choices = 21
            args.joint_cluster_choice_ranks = ranks
            args.joint_cluster_choice_sizes = sizes
            args.joint_num_ap_tokens = 5
            args.joint_num_actions = 43
            args.joint_valid_action_indices = list(range(21)) + list(range(21, 42)) + [42]
            action_space = spaces.MultiDiscrete([43])
            actor = R_Actor(args, self.obs_space, action_space)
            actor.eval()
            with torch.no_grad():
                (actions, log_probs, _) = actor(self.obs, self.rnn, self.masks, deterministic=False)
            self.assertTrue(actor._structured_subset)
            self.assertEqual(tuple(actions.shape), (5, 1))
            self.assertEqual(tuple(log_probs.shape), (5, 1))
            self.assertTrue(torch.all(actions < 43))

    def test_conditional_subset_actor_has_mode_specific_choices(self):
        from itertools import combinations
        ranks = [list(choice) for size in (2, 3, 5) for choice in combinations(range(5), size)]
        sizes = [size for size in (2, 3, 5) for _choice in combinations(range(5), size)]
        for backbone in ('mlp', 'transformer'):
            args = actor_args(backbone)
            args.joint_action_head = 'subset_conditional'
            args.joint_num_cluster_choices = 21
            args.joint_cluster_choice_ranks = ranks
            args.joint_cluster_choice_sizes = sizes
            args.joint_num_ap_tokens = 5
            args.joint_num_actions = 43
            args.joint_valid_action_indices = list(range(21)) + list(range(21, 42)) + [42]
            actor = R_Actor(args, self.obs_space, spaces.MultiDiscrete([43]))
            actor.eval()
            with torch.no_grad():
                (actions, log_probs, _) = actor(self.obs, self.rnn, self.masks, deterministic=False)
            self.assertEqual(tuple(actions.shape), (5, 1))
            self.assertEqual(tuple(log_probs.shape), (5, 1))
            self.assertEqual(tuple(actor.act.last_choice_probs.shape), (5, 2, 21))

    def test_native_mlp_actor_uses_flat_observation_and_direct_head(self):
        args = actor_args('mlp')
        args.joint_action_head = 'flat'
        args.actor_token_norm = False
        actor = R_Actor(args, self.obs_space, spaces.MultiDiscrete([7]))
        self.assertFalse(actor._structured_subset)
        self.assertFalse(actor._gated_joint)
        self.assertIsInstance(actor.act, FlatJointACTLayer)
        actor.eval()
        with torch.no_grad():
            (actions, log_probs, _) = actor(self.obs, self.rnn, self.masks, deterministic=True)
        self.assertEqual(tuple(actions.shape), (5, 1))
        self.assertEqual(tuple(log_probs.shape), (5, 1))
        self.assertEqual(tuple(actor.base.feature_norm.normalized_shape), (24,))

    def test_native_mlp_actor_uses_factorized_subset_head(self):
        from itertools import combinations
        ranks = [list(choice) for size in (2, 3, 4, 5) for choice in combinations(range(5), size)]
        sizes = [size for size in (2, 3, 4, 5) for _choice in combinations(range(5), size)]
        args = actor_args('mlp')
        args.joint_action_head = 'mlp_factorized'
        args.actor_token_norm = False
        args.joint_num_cluster_choices = 26
        args.joint_cluster_choice_ranks = ranks
        args.joint_cluster_choice_sizes = sizes
        args.joint_num_ap_tokens = 5
        args.joint_num_actions = 53
        args.joint_valid_action_indices = list(range(26)) + list(range(26, 52)) + [52]
        actor = R_Actor(args, self.obs_space, spaces.MultiDiscrete([53]))
        self.assertTrue(actor._structured_subset)
        self.assertFalse(actor._score_ap_tokens)
        self.assertIsInstance(actor.act, StructuredSubsetACTLayer)
        self.assertFalse(actor.act.score_from_tokens)
        self.assertIsNone(actor.subset_token_encoder)
        actor.eval()
        with torch.no_grad():
            (actions, log_probs, _) = actor(self.obs, self.rnn, self.masks, deterministic=True)
        self.assertEqual(tuple(actions.shape), (5, 1))
        self.assertEqual(tuple(log_probs.shape), (5, 1))

    def test_transformer_native_actor_uses_relational_subset_head(self):
        from itertools import combinations
        ranks = [list(choice) for size in (2, 3, 4, 5) for choice in combinations(range(5), size)]
        sizes = [size for size in (2, 3, 4, 5) for _choice in combinations(range(5), size)]
        args = actor_args('transformer')
        args.joint_action_head = 'relational_subset'
        args.joint_num_cluster_choices = 26
        args.joint_cluster_choice_ranks = ranks
        args.joint_cluster_choice_sizes = sizes
        args.joint_num_ap_tokens = 5
        args.joint_num_actions = 53
        args.joint_valid_action_indices = list(range(26)) + list(range(26, 52)) + [52]
        actor = R_Actor(args, self.obs_space, spaces.MultiDiscrete([53]))
        self.assertTrue(actor._structured_subset)
        self.assertIsInstance(actor.act, RelationalSubsetACTLayer)
        actor.eval()
        with torch.no_grad():
            (actions, log_probs, _) = actor(self.obs, self.rnn, self.masks, deterministic=True)
        self.assertEqual(tuple(actions.shape), (5, 1))
        self.assertEqual(tuple(log_probs.shape), (5, 1))

    def test_relational_ap_head_uses_cross_ap_context(self):
        from itertools import combinations
        ranks = [list(choice) for size in (2, 3, 4, 5) for choice in combinations(range(5), size)]
        sizes = [size for size in (2, 3, 4, 5) for _choice in combinations(range(5), size)]
        layer = RelationalSubsetACTLayer(spaces.MultiDiscrete([53]), inputs_dim=16, num_modes=3, num_choices=26, cluster_choice_ranks=ranks, cluster_choice_sizes=sizes, num_ap_tokens=5, valid_action_indices=list(range(26)) + list(range(26, 52)) + [52], conditional_modes=True, actor_temporal_context='history_mean', tokens_per_frame=6, relation_dim=8)
        pooled = torch.randn(2, 16)
        tokens = torch.randn(2, 24, 16)
        changed = tokens.clone()
        changed[:, -5, :] += 2.0
        with torch.no_grad():
            original = layer.joint_logits(pooled, tokens=tokens)
            relational = layer.joint_logits(pooled, tokens=changed)
        self.assertEqual(tuple(original.shape), (2, 53))
        self.assertFalse(torch.allclose(original, relational))

    def test_mlp_actor_can_normalize_structured_tokens_separately(self):
        args = actor_args('mlp')
        args.actor_token_norm = True
        actor = R_Actor(args, self.obs_space, spaces.MultiDiscrete([7]))
        self.assertEqual(tuple(actor.base.feature_norm.normalized_shape), (4,))
        with torch.no_grad():
            (actions, log_probs, _) = actor(self.obs, self.rnn, self.masks, deterministic=True)
        self.assertEqual(tuple(actions.shape), (5, 1))
        self.assertEqual(tuple(log_probs.shape), (5, 1))

    def test_current_frame_context_ignores_stale_pooled_history(self):
        from itertools import combinations
        ranks = [list(choice) for size in (2, 3, 4, 5) for choice in combinations(range(5), size)]
        sizes = [size for size in (2, 3, 4, 5) for _choice in combinations(range(5), size)]
        layer = StructuredSubsetACTLayer(spaces.MultiDiscrete([53]), inputs_dim=16, num_modes=3, num_choices=26, cluster_choice_ranks=ranks, cluster_choice_sizes=sizes, num_ap_tokens=5, valid_action_indices=list(range(26)) + list(range(26, 52)) + [52], conditional_modes=True, actor_temporal_context='current_frame', tokens_per_frame=6)
        current_frame = torch.randn(2, 6, 16)
        tokens_a = torch.cat((torch.zeros(2, 18, 16), current_frame), dim=1)
        tokens_b = torch.cat((torch.ones(2, 18, 16), current_frame), dim=1)
        pooled_a = torch.zeros(2, 16)
        pooled_b = torch.ones(2, 16)
        with torch.no_grad():
            logits_a = layer.joint_logits(pooled_a, tokens=tokens_a)
            logits_b = layer.joint_logits(pooled_b, tokens=tokens_b)
        self.assertTrue(torch.allclose(logits_a, logits_b, atol=1e-06))

    def test_balanced_current_frame_weights_ue_and_ap_equally(self):
        from itertools import combinations
        ranks = [list(choice) for size in (2, 3, 4, 5) for choice in combinations(range(5), size)]
        sizes = [size for size in (2, 3, 4, 5) for _choice in combinations(range(5), size)]
        layer = StructuredSubsetACTLayer(spaces.MultiDiscrete([53]), inputs_dim=16, num_modes=3, num_choices=26, cluster_choice_ranks=ranks, cluster_choice_sizes=sizes, num_ap_tokens=5, valid_action_indices=list(range(26)) + list(range(26, 52)) + [52], conditional_modes=True, actor_temporal_context='current_frame_balanced', tokens_per_frame=6)
        current_frame = torch.randn(2, 6, 16)
        tokens = torch.randn(2, 24, 16)
        tokens[:, -6:, :] = current_frame
        context = layer._context_features(torch.zeros(2, 16), tokens)
        expected = 0.5 * (current_frame[:, 0, :] + current_frame[:, 1:, :].mean(dim=1))
        self.assertTrue(torch.allclose(context, expected, atol=1e-06))
if __name__ == '__main__':
    unittest.main()
