"""IICMS mobility MAPPO training entry point."""
import sys
import os
import socket
import hashlib
from utils.runtime_env import normalize_thread_environment
normalize_thread_environment()
try:
    import setproctitle
except Exception:
    setproctitle = None
import numpy as np
from pathlib import Path
import torch
import json
from config import get_config
from envs.env_wrappers import make_vec_env

def make_train_env(all_args):

    def get_env_fn(rank):

        def init_env():
            if all_args.action_space == 'continuous':
                from envs.env_continuous import ContinuousActionEnv as EnvCls
            else:
                from envs.env_iicms_mobile import IICMSMobileDiscreteActionEnv as EnvCls
            env = EnvCls()
            if hasattr(env, 'configure'):
                env.configure(all_args)
            if getattr(all_args, 'channel_file', None) and hasattr(env, 'load_channel'):
                try:
                    env.load_channel(all_args.channel_file)
                except Exception as e:
                    print(f'[WARN] load_channel failed on rank={rank}: {e}')
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    return make_vec_env([get_env_fn(i) for i in range(all_args.n_rollout_threads)], getattr(all_args, 'vec_env', 'auto'))

def make_eval_env(all_args):

    def get_env_fn(rank):

        def init_env():
            if all_args.action_space == 'continuous':
                from envs.env_continuous import ContinuousActionEnv as EnvCls
            else:
                from envs.env_iicms_mobile import IICMSMobileDiscreteActionEnv as EnvCls
            env = EnvCls()
            if hasattr(env, 'configure'):
                env.configure(all_args)
            if getattr(all_args, 'channel_file', None) and hasattr(env, 'load_channel'):
                try:
                    env.load_channel(all_args.channel_file)
                except Exception as e:
                    print(f'[WARN] load_channel (eval) failed on rank={rank}: {e}')
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    return make_vec_env([get_env_fn(0)], 'dummy')

def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str, default='MyEnv', help='Which scenario to run on')
    parser.add_argument('--num_landmarks', type=int, default=3)
    parser.add_argument('--num_agents', type=int, default=30, help='number of players')
    all_args = parser.parse_known_args(args)[0]
    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    if all_args.original_mappo:
        all_args.share_policy = False
        all_args.use_centralized_V = True
        all_args.critic_dedup = False
        all_args.actor_backbone = 'mlp'
        all_args.critic_backbone = 'mlp'
        all_args.actor_token_norm = False
        all_args.joint_action_head = 'paper_factorized'
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
        print('[algorithm] original_mappo: independent UE actors, concatenated-state critics, factorized mode x cluster-size', flush=True)
    if not 0.0 <= all_args.backhaul_lo <= all_args.backhaul_hi:
        raise ValueError(f'Require 0 <= backhaul_lo <= backhaul_hi, got {all_args.backhaul_lo} and {all_args.backhaul_hi}')
    if int(all_args.validation_interval_steps) <= 0:
        raise ValueError('validation_interval_steps must be positive')
    if int(all_args.validation_episodes) <= 0:
        raise ValueError('validation_episodes must be positive')
    if int(all_args.validation_horizon) < 0:
        raise ValueError('validation_horizon must be non-negative')
    if float(all_args.validation_plateau_relative_gain) < 0.0:
        raise ValueError('validation_plateau_relative_gain must be non-negative')
    if int(all_args.validation_window) < 3:
        raise ValueError('validation_window must be at least 3')
    if all_args.reward_mode == 'mean_se' and int(all_args.objective_horizon) != int(all_args.episode_length):
        raise ValueError(f'The time-averaged SE objective requires objective_horizon to equal episode_length; got {all_args.objective_horizon} and {all_args.episode_length}')
    if all_args.algorithm_name == 'rmappo':
        assert all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy, 'check recurrent policy!'
    elif all_args.algorithm_name == 'mappo':
        assert all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False, 'check recurrent policy!'
    else:
        raise NotImplementedError
    if all_args.cuda and torch.cuda.is_available():
        print('choose to use gpu...')
        device = torch.device('cuda:0')
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print('choose to use cpu...')
        device = torch.device('cpu')
        torch.set_num_threads(all_args.n_training_threads)
    project_root = Path(__file__).resolve().parent
    results_root = project_root / 'results'
    run_dir = results_root / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    os.makedirs(run_dir, exist_ok=True)
    exst_run_nums = [int(f.name.split('run')[1]) for f in run_dir.iterdir() if f.name.startswith('run')]
    curr_run = 'run1' if len(exst_run_nums) == 0 else f'run{max(exst_run_nums) + 1}'
    run_dir = run_dir / curr_run
    os.makedirs(str(run_dir), exist_ok=True)
    project_root_path = Path(__file__).resolve().parent
    tracked_sources = ['config.py', 'evaluate_iicms.py', 'envs/env_core.py', 'envs/env_iicms_mobile.py', 'envs/env_wrappers.py', 'algorithms/algorithm/r_actor_critic.py', 'algorithms/algorithm/r_mappo.py', 'algorithms/algorithm/rMAPPOPolicy.py', 'algorithms/utils/transformer.py', 'algorithms/utils/act.py', 'algorithms/utils/popart.py', 'utils/numpy_torch.py', 'utils/runtime_env.py', 'utils/valuenorm.py', 'utils/shared_buffer.py', 'utils/separated_buffer.py', 'utils/policy_evaluation.py', 'utils/util.py', 'algorithms/utils/mlp.py', 'runner/shared/env_runner.py', 'runner/shared/base_runner.py', 'runner/separated/env_runner.py', 'runner/separated/base_runner.py']
    source_hashes = {}
    for rel in tracked_sources:
        path = project_root_path / rel
        if path.exists():
            source_hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    with open(run_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(vars(all_args), f, ensure_ascii=False, indent=2)
    with open(run_dir / 'source_sha256.json', 'w', encoding='utf-8') as f:
        json.dump(source_hashes, f, ensure_ascii=False, indent=2)
    try:
        if setproctitle is not None:
            setproctitle.setproctitle(f'{all_args.algorithm_name}-{all_args.env_name}-{all_args.experiment_name}@{all_args.user_name}')
    except Exception:
        pass
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents
    worker_metadata = getattr(envs, 'worker_metadata', {})
    env0 = envs.envs[0] if hasattr(envs, 'envs') else None
    all_args.actor_token_dim = int(worker_metadata.get('obs_token_dim', getattr(env0, 'obs_token_dim', 1)))
    all_args.critic_token_dim = int(worker_metadata.get('obs_dim', getattr(env0, 'obs_dim', envs.observation_space[0].shape[0])))
    all_args.joint_num_modes = int(worker_metadata.get('joint_num_modes', getattr(env0, 'joint_num_modes', 3)))
    all_args.joint_num_sizes = int(worker_metadata.get('joint_num_sizes', getattr(env0, 'joint_num_sizes', 3)))
    all_args.joint_num_actions = int(envs.action_space[0].nvec[0])
    all_args.joint_valid_action_indices = list(worker_metadata.get('joint_action_indices', getattr(env0, 'joint_action_indices', list(range(all_args.joint_num_actions)))))
    all_args.joint_action_names = list(worker_metadata.get('joint_action_names', getattr(env0, 'joint_action_names', [])))
    all_args.joint_num_cluster_choices = int(worker_metadata.get('joint_num_cluster_choices', getattr(env0, 'joint_num_cluster_choices', 0)))
    all_args.joint_cluster_choice_ranks = list(worker_metadata.get('joint_cluster_choice_ranks', getattr(env0, 'joint_cluster_choice_ranks', [])))
    all_args.joint_cluster_choice_sizes = list(worker_metadata.get('joint_cluster_choice_sizes', getattr(env0, 'joint_cluster_choice_sizes', [])))
    all_args.joint_action_cluster_choice = list(worker_metadata.get('joint_action_cluster_choice', getattr(env0, 'joint_action_cluster_choice', [])))
    all_args.joint_num_ap_tokens = int(worker_metadata.get('joint_num_ap_tokens', getattr(env0, 'joint_num_ap_tokens', 0)))
    if len(all_args.joint_valid_action_indices) != all_args.joint_num_actions:
        raise ValueError(f'Joint action metadata mismatch: {all_args.joint_num_actions} actions vs {len(all_args.joint_valid_action_indices)} valid indices')
    with open(run_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump({**vars(all_args), 'resolved_obs_dim': int(envs.observation_space[0].shape[0]), 'resolved_share_obs_dim': int(envs.share_observation_space[0].shape[0]), 'resolved_action_nvec': [int(v) for v in envs.action_space[0].nvec], 'resolved_joint_action_names': worker_metadata.get('joint_action_names', getattr(env0, 'joint_action_names', [])), 'resolved_joint_action_indices': list(all_args.joint_valid_action_indices), 'resolved_actor_token_dim': all_args.actor_token_dim, 'resolved_critic_token_dim': all_args.critic_token_dim}, f, ensure_ascii=False, indent=2)
    config = {'all_args': all_args, 'envs': envs, 'eval_envs': eval_envs, 'num_agents': num_agents, 'device': device, 'run_dir': run_dir}
    if all_args.share_policy:
        from runner.shared.env_runner import EnvRunner as Runner
    else:
        from runner.separated.env_runner import EnvRunner as Runner
    runner = Runner(config)
    try:
        runner.run()
    finally:
        try:
            envs.close()
            if all_args.use_eval and eval_envs is not envs:
                eval_envs.close()
        finally:
            summary_path = Path(runner.log_dir) / 'summary.json' if hasattr(runner, 'log_dir') else run_dir / 'logs/summary.json'
            if hasattr(runner, 'writter'):
                try:
                    runner.writter.export_scalars_to_json(str(summary_path))
                    runner.writter.close()
                except Exception:
                    pass
if __name__ == '__main__':
    main(sys.argv[1:])
