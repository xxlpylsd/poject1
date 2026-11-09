from __future__ import annotations
import sys
import unittest
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy
from config import get_config
from envs.env_iicms_mobile import IICMSMobileDiscreteActionEnv
from train_main import parse_args

def build_policy(index: int=0) -> tuple[RMAPPOPolicy, object]:
    parser = get_config()
    args = parse_args(['--num_agents', '2', '--num_aps', '12', '--cluster_sizes', '2,3,4', '--cluster_action_mode', 'subset', '--obs_frames', '1', '--obs_include_xy', '1', '--original_mappo', '--seed', str(100 + index)], parser)
    args.share_policy = False
    args.use_centralized_V = True
    args.actor_backbone = 'mlp'
    args.critic_backbone = 'mlp'
    args.joint_action_head = 'paper_factorized'
    args.use_recurrent_policy = False
    args.use_naive_recurrent_policy = False
    args.hidden_size = 32
    args.critic_hidden_size = 32
    env = IICMSMobileDiscreteActionEnv()
    env.configure(args)
    args.actor_token_dim = env.obs_token_dim
    args.critic_token_dim = env.signal_obs_dim
    args.joint_num_modes = env.joint_num_modes
    args.joint_num_sizes = env.joint_num_sizes
    args.joint_num_actions = env.joint_num_actions
    args.joint_valid_action_indices = list(env.joint_action_indices)
    args.joint_num_cluster_choices = env.joint_num_cluster_choices
    args.joint_cluster_choice_ranks = [list(ranks) for ranks in env.joint_cluster_choice_ranks]
    args.joint_cluster_choice_sizes = list(env.joint_cluster_choice_sizes)
    args.joint_action_cluster_choice = list(env.joint_action_cluster_choice)
    args.joint_num_ap_tokens = env.joint_num_ap_tokens
    policy = RMAPPOPolicy(args, env.observation_space[0], env.share_observation_space[0], env.action_space[0], device=torch.device('cpu'))
    return (policy, env)

class OriginalMAPPOTest(unittest.TestCase):

    def test_agents_have_independent_actor_and_critic_parameters(self):
        (first, _env) = build_policy(0)
        (second, _env2) = build_policy(1)
        first_ptrs = {parameter.data_ptr() for parameter in first.actor.parameters()}
        second_ptrs = {parameter.data_ptr() for parameter in second.actor.parameters()}
        self.assertTrue(first_ptrs.isdisjoint(second_ptrs))
        self.assertIsNot(first.critic.base.mlp.fc1[0].weight, second.critic.base.mlp.fc1[0].weight)

    def test_factorized_action_maps_only_to_top_l_branches(self):
        (policy, env) = build_policy()
        obs = torch.randn(4, env.signal_obs_dim)
        rnn = torch.zeros(4, 1, 1)
        masks = torch.ones(4, 1)
        (action, log_prob, _rnn) = policy.actor(obs, rnn, masks, deterministic=True)
        self.assertEqual(tuple(action.shape), (4, 1))
        self.assertEqual(tuple(log_prob.shape), (4, 1))
        for value in action.view(-1).tolist():
            name = env.joint_action_names[int(value)]
            self.assertTrue(name == 'NC' or name.endswith('_S00'), msg=name)

    def test_factorized_entropy_reaches_both_heads(self):
        (policy, env) = build_policy()
        obs = torch.randn(6, env.signal_obs_dim)
        action = torch.randint(0, env.joint_num_actions, (6, 1))
        rnn = torch.zeros(6, 1, 1)
        masks = torch.ones(6, 1)
        (_log_prob, entropy) = policy.actor.evaluate_actions(obs, rnn, action, masks)
        entropy.backward()
        self.assertIsNotNone(policy.actor.act.mode_head.weight.grad)
        self.assertIsNotNone(policy.actor.act.size_head.weight.grad)
        self.assertGreater(float(policy.actor.act.mode_head.weight.grad.abs().sum() + policy.actor.act.size_head.weight.grad.abs().sum()), 0.0)
if __name__ == '__main__':
    unittest.main()
