"""Select checkpoints on fixed validation episodes."""
from __future__ import annotations
import argparse
import json
import re
import shutil
from argparse import Namespace
from pathlib import Path
from utils.runtime_env import normalize_thread_environment
normalize_thread_environment()
import torch
from algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy
from envs.env_iicms_mobile import IICMSMobileDiscreteActionEnv
from utils.policy_evaluation import SharedPolicyActor, evaluate_episodes

def snapshot_steps(models_dir: Path) -> list[int]:
    steps = []
    for path in models_dir.glob('actor_step_*.pt'):
        match = re.fullmatch('actor_step_(\\d+)\\.pt', path.name)
        if match:
            steps.append(int(match.group(1)))
    return sorted(set(steps))

def load_config(run_dir: Path) -> Namespace:
    config = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    config.setdefault('cluster_sizes', '2,3,4,5')
    config.setdefault('obs_include_xy', 0)
    config.setdefault('obs_frames', 1)
    return Namespace(**config)

def make_policy(all_args: Namespace, env, device: torch.device):
    all_args.actor_token_dim = env.obs_token_dim
    all_args.critic_token_dim = env.signal_obs_dim
    all_args.joint_num_modes = env.joint_num_modes
    all_args.joint_num_sizes = env.joint_num_sizes
    all_args.joint_num_actions = env.joint_num_actions
    all_args.joint_valid_action_indices = list(env.joint_action_indices)
    all_args.joint_num_cluster_choices = env.env.joint_num_cluster_choices
    all_args.joint_cluster_choice_ranks = [list(ranks) for ranks in env.env.cluster_choice_ranks]
    all_args.joint_cluster_choice_sizes = list(env.env.cluster_choice_sizes)
    all_args.joint_action_cluster_choice = list(env.env.joint_action_cluster_choice)
    all_args.joint_num_ap_tokens = env.obs_tokens_per_frame - 1
    return RMAPPOPolicy(all_args, env.observation_space[0], env.share_observation_space[0], env.action_space[0], device=device)

def evaluate_candidate(policy, env, all_args: Namespace, actor_path: Path, critic_path: Path, seeds: list[int], horizon: int, qos_weight: float) -> dict:
    policy.actor.load_state_dict(torch.load(actor_path, map_location=policy.device))
    if critic_path.exists():
        policy.critic.load_state_dict(torch.load(critic_path, map_location=policy.device))
    metrics = evaluate_episodes(env, all_args, seeds, horizon, SharedPolicyActor(policy, policy.device, env.num_agent))
    mean_ue_se = float(metrics['mean_ue_se_bps_hz'])
    violation_rate = float(metrics['qos_violation_rate'])
    shortfall_mbps = float(metrics['qos_shortfall_mbps'])
    requirement = max(float(env.env.rate_req_lo_mbps), 1e-12)
    normalized_shortfall = shortfall_mbps / requirement
    qos_aware_score = mean_ue_se - qos_weight * (violation_rate + normalized_shortfall)
    return {'system_se_bps_hz': float(metrics['system_se_bps_hz']), 'mean_ue_se_bps_hz': mean_ue_se, 'p05_mbps': float(metrics['p05_rate_mbps']), 'p50_mbps': float(metrics['p50_rate_mbps']), 'qos_violation_rate': violation_rate, 'qos_shortfall_mbps': shortfall_mbps, 'overload_ratio': float(metrics['overload_ratio']), 'fronthaul_drop_ratio': float(metrics['fronthaul_drop_ratio']), 'fronthaul_queue_ratio': float(metrics['fronthaul_queue_ratio']), 'qos_aware_score': float(qos_aware_score)}

def resolve_selection_metric(metric: str, qos_penalty_weight: float) -> str:
    if metric != 'auto':
        return metric
    if float(qos_penalty_weight) > 0.0:
        return 'qos_constrained'
    return 'system_se'

def select_candidate(candidates: list[dict], metric: str, max_qos_violation_rate: float) -> tuple[dict, str]:
    """Select a checkpoint with the requested validation objective."""
    if not candidates:
        raise ValueError('No checkpoint candidates were provided')
    if metric == 'system_se':
        best = max(candidates, key=lambda item: (item['system_se_bps_hz'], -item['qos_violation_rate']))
        return (best, 'system_se')
    if metric == 'qos_aware':
        best = max(candidates, key=lambda item: (item['qos_aware_score'], item['system_se_bps_hz'], -item['qos_violation_rate']))
        return (best, 'qos_aware')
    if metric != 'qos_constrained':
        raise ValueError(f'Unknown selection metric: {metric}')
    feasible = [item for item in candidates if item['qos_violation_rate'] <= max_qos_violation_rate + 1e-12]
    if feasible:
        return (max(feasible, key=lambda item: (item['system_se_bps_hz'], -item['qos_shortfall_mbps'])), 'qos_constrained')
    minimum_violation = min((item['qos_violation_rate'] for item in candidates))
    least_infeasible = [item for item in candidates if item['qos_violation_rate'] <= minimum_violation + 1e-12]
    return (max(least_infeasible, key=lambda item: item['system_se_bps_hz']), 'qos_constrained_fallback')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--seed_start', type=int, default=91000)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--metric', choices=('auto', 'qos_constrained', 'qos_aware', 'system_se'), default='auto', help='auto uses system_se for pure-rate training and qos_constrained when a positive QoS penalty is configured')
    parser.add_argument('--max_qos_violation_rate', type=float, default=0.005, help='maximum validation QoS violation rate for qos_constrained checkpoint selection')
    parser.add_argument('--qos_penalty_weight', type=float, default=20.0)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    models_dir = run_dir / 'models'
    all_args = load_config(run_dir)
    args.metric = resolve_selection_metric(args.metric, getattr(all_args, 'qos_penalty_weight', 0.0))
    env = IICMSMobileDiscreteActionEnv()
    env.configure(all_args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    policy = make_policy(all_args, env, device)
    seeds = [args.seed_start + index for index in range(args.episodes)]
    steps = snapshot_steps(models_dir)
    candidates = []
    if steps:
        for step in steps:
            actor_path = models_dir / f'actor_step_{step:08d}.pt'
            critic_path = models_dir / f'critic_step_{step:08d}.pt'
            metrics = evaluate_candidate(policy, env, all_args, actor_path, critic_path, seeds, args.horizon, args.qos_penalty_weight)
            candidates.append({'step': step, 'actor': actor_path.name, 'critic': critic_path.name, **metrics})
    else:
        candidates.append({'step': None, 'actor': 'actor.pt', 'critic': 'critic.pt', **evaluate_candidate(policy, env, all_args, models_dir / 'actor.pt', models_dir / 'critic.pt', seeds, args.horizon, args.qos_penalty_weight)})
    final_actor = models_dir / 'actor_final.pt'
    final_critic = models_dir / 'critic_final.pt'
    if not final_actor.exists():
        shutil.copyfile(models_dir / 'actor.pt', final_actor)
    if not final_critic.exists():
        shutil.copyfile(models_dir / 'critic.pt', final_critic)
    if not 0.0 <= args.max_qos_violation_rate <= 1.0:
        parser.error('--max_qos_violation_rate must be in [0, 1]')
    (best, selection_status) = select_candidate(candidates, args.metric, args.max_qos_violation_rate)
    best_actor = models_dir / best['actor']
    best_critic = models_dir / best['critic']
    shutil.copyfile(best_actor, models_dir / 'actor_best.pt')
    shutil.copyfile(best_critic, models_dir / 'critic_best.pt')
    shutil.copyfile(models_dir / 'actor_best.pt', models_dir / 'actor.pt')
    shutil.copyfile(models_dir / 'critic_best.pt', models_dir / 'critic.pt')
    result = {'run_dir': str(run_dir), 'selection_seed_start': args.seed_start, 'selection_episodes': args.episodes, 'selection_horizon': args.horizon, 'selection_metric': args.metric, 'max_qos_violation_rate': args.max_qos_violation_rate, 'selection_status': selection_status, 'scenario_seed_policy': 'selection_episode_seed', 'selected_step': best['step'], 'selected': best, 'candidates': candidates}
    (run_dir / 'checkpoint_selection.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
