from __future__ import annotations
import argparse
import hashlib
import json
import time
from argparse import Namespace
from pathlib import Path
from utils.runtime_env import normalize_thread_environment
normalize_thread_environment()
import numpy as np
import torch
from algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy
from envs.env_iicms_mobile import IICMSMobileDiscreteActionEnv
ENV_PROTOCOL_KEYS = ('env_name', 'scenario_name', 'num_agents', 'num_aps', 'antennas_per_ap', 'map_size', 'ap_min_distance', 'backhaul_lo', 'backhaul_hi', 'backhaul_jt_model', 'fronthaul_queue', 'fronthaul_buffer_s', 'cluster_sizes', 'cluster_action_mode', 'cluster_policy', 'mobility', 'ue_speed', 'slot_dt', 'csi_aging', 'carrier_ghz', 'csi_block_slots', 'csi_pilot_slots', 'csi_slot_ms', 'pathloss_model', 'obs_include_xy', 'obs_frames', 'obs_neighbor_ues', 'obs_ap_competition', 'objective_horizon', 'rate_req_lo_mbps', 'rate_req_hi_mbps', 'min_ue_se_bps_hz', 'qos_penalty_weight', 'reward_mode', 'reward_phi1', 'reward_phi2', 'joint_action_head')

def encode_action(env, action: np.ndarray) -> np.ndarray:
    action_index = int(np.asarray(action).reshape(-1)[0])
    encoded = np.zeros(len(env.env.joint_actions), dtype=np.float32)
    encoded[action_index] = 1.0
    return encoded

def empirical_cdf(values: np.ndarray, points: int=501) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(values, dtype=float).reshape(-1)
    if samples.size == 0:
        raise ValueError('Cannot build a CDF from zero samples')
    if not np.all(np.isfinite(samples)):
        raise ValueError('CDF samples must be finite')
    upper = float(np.max(samples))
    if upper <= 0.0:
        upper = 1.0
    grid = np.linspace(0.0, upper, max(2, int(points)))
    ordered = np.sort(samples)
    cdf = np.searchsorted(ordered, grid, side='right').astype(float) / float(samples.size)
    cdf[0] = float(np.mean(samples <= 0.0))
    cdf[-1] = 1.0
    return (grid, cdf)

def seed_range(seed_start: int, episodes: int) -> tuple[int, int]:
    if episodes < 1:
        raise ValueError('episodes must be positive')
    return (seed_start, seed_start + episodes - 1)

def ranges_overlap(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return max(first[0], second[0]) <= min(first[1], second[1])

def resolve_fixed_action(env, requested: str) -> tuple[str, int]:
    normalized = requested.strip().upper().replace('-', '_')
    aliases = {'CB_L5': 'CB_L5_S00', 'CB_L5_TOP': 'CB_L5_S00', 'JT_L2': 'JT_L2_S00', 'JT_L3': 'JT_L3_S00', 'JT_L5': 'JT_L5_S00'}
    wanted = aliases.get(normalized, normalized)
    names = [str(name).upper() for name in env.joint_action_names]
    if wanted in names:
        index = names.index(wanted)
        return (env.joint_action_names[index], index)
    matches = [(name, index) for (index, name) in enumerate(names) if name == wanted or name.startswith(f'{wanted}_')]
    if len(matches) == 1:
        return (env.joint_action_names[matches[0][1]], matches[0][1])
    available = ', '.join(env.joint_action_names)
    raise ValueError(f'Unknown fixed action {requested!r}; normalized={wanted!r}. Available actions: {available}')

def rssi_modes(env, threshold_hi: float, threshold_lo: float) -> np.ndarray:
    if threshold_lo <= 0.0 or threshold_hi < threshold_lo:
        raise ValueError('RSSI thresholds must satisfy 0 < low <= high')
    order = env.env._candidate_ap_order()
    primary = order[:, 0]
    best = env.env.mu[np.arange(env.num_agent), primary]
    other = np.maximum(np.sum(env.env.mu, axis=1) - best, 1e-30)
    ratio = best / other
    modes = np.full(env.num_agent, 2, dtype=np.int64)
    modes[ratio <= threshold_hi] = 1
    modes[ratio <= threshold_lo] = 0
    return modes

def rssi_action_indices(env, cluster_size: int) -> dict[int, int]:
    if int(cluster_size) not in env.env.size_options:
        raise ValueError(f'RSSI cluster size {cluster_size} is not in configured sizes {env.env.size_options}')
    return {0: resolve_fixed_action(env, f'JT_L{int(cluster_size)}_S00')[1], 1: resolve_fixed_action(env, f'CB_L{int(cluster_size)}_S00')[1], 2: resolve_fixed_action(env, 'NC')[1]}

def load_checkpoint_selection(run_dir: Path) -> dict | None:
    path = run_dir / 'checkpoint_selection.json'
    if not path.exists():
        return None
    selection = json.loads(path.read_text(encoding='utf-8'))
    required = ('selection_seed_start', 'selection_episodes')
    missing = [key for key in required if key not in selection]
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(missing)}")
    return selection

def checkpoint_path(run_dir: Path, role: str) -> Path:
    names = {'actor': 'actor.pt', 'best': 'actor_best.pt', 'final': 'actor_final.pt'}
    path = run_dir / 'models' / names[role]
    if not path.exists():
        raise FileNotFoundError(f'Requested checkpoint {role!r} does not exist: {path}. Use --checkpoint actor for legacy runs without snapshots.')
    return path

def checkpoint_paths(run_dir: Path, role: str, num_agents: int, separate_policy: bool) -> list[Path]:
    if not separate_policy:
        return [checkpoint_path(run_dir, role)]
    if role != 'actor':
        raise ValueError('The original-MAPPO comparison runner stores final per-UE checkpoints only; use --checkpoint actor and disable snapshot-based checkpoint selection.')
    paths = [run_dir / 'models' / f'actor_agent{agent_id}.pt' for agent_id in range(num_agents)]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError('Missing original-MAPPO actor files: ' + ', '.join((str(path) for path in missing)))
    return paths

def canonical_env_config(config: dict) -> dict:
    return {key: config.get(key) for key in ENV_PROTOCOL_KEYS if key in config}

def initial_state_digest(env) -> str:
    core = env.env
    fields = ('users_xy', 'aps_xy', 'backhaul_caps', 'mu', '_shadow_gain', '_rate_req_mbps', '_waypoints', '_velocity', '_speed')
    digest = hashlib.sha256()
    for name in fields:
        value = getattr(core, name)
        digest.update(name.encode('ascii'))
        digest.update(np.asarray(value, dtype=np.float64).tobytes(order='C'))
    return digest.hexdigest()

def protocol_id(config: dict, seeds: list[int], horizon: int, phase: str, state_digests: list[str]) -> str:
    payload = {'phase': phase, 'evaluation_seeds': seeds, 'horizon': horizon, 'scenario_seed_policy': 'evaluation_episode_seed', 'initial_state_digests': state_digests, 'environment': canonical_env_config(config)}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--seed_start', type=int, default=93000, help='first evaluation seed; use 91000 for hyperparameter selection')
    parser.add_argument('--phase', choices=('validation', 'test', 'diagnostic'), default='test', help='test refuses seeds overlapping checkpoint selection; use validation only for model selection')
    parser.add_argument('--checkpoint', choices=('actor', 'best', 'final'), default='best', help='learned checkpoint to evaluate')
    parser.add_argument('--fixed_action', default=None, help='evaluate a fixed action such as CB_L5_S00 instead of a learned policy')
    parser.add_argument('--baseline_policy', choices=('fixed', 'rssi'), default=None, help='fixed action or RSSI-threshold adaptive mode selection')
    parser.add_argument('--cluster_policy', choices=('static', 'dynamic'), default=None, help='static freezes the candidate AP ranking for one episode; dynamic refreshes it at every decision step')
    parser.add_argument('--rssi_cluster_size', type=int, default=None, help='cooperative cluster size used by the RSSI baseline')
    parser.add_argument('--rssi_threshold_hi', type=float, default=20.0)
    parser.add_argument('--rssi_threshold_lo', type=float, default=1.0)
    parser.add_argument('--policy_label', default=None, help='stable label written into the evaluation JSON')
    parser.add_argument('--allow_seed_overlap', action='store_true', help='explicitly allow test seeds used during checkpoint selection')
    parser.add_argument('--stochastic', action='store_true', help='sample from the policy instead of using deterministic actions')
    parser.add_argument('--num_aps_override', type=int, default=None, help='evaluate the same trained model with a different number of APs; policy input must remain shape-compatible')
    parser.add_argument('--backhaul_lo_override', type=float, default=None, help='override the lower per-AP backhaul capacity in bit/s')
    parser.add_argument('--backhaul_hi_override', type=float, default=None, help='override the upper per-AP backhaul capacity in bit/s')
    parser.add_argument('--out', default='results_eval_iicms')
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    config_path = run_dir / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    config.setdefault('cluster_sizes', '2,3,5')
    config.setdefault('cluster_policy', 'dynamic')
    if args.num_aps_override is not None:
        if int(args.num_aps_override) < 2:
            parser.error('--num_aps_override must be at least 2')
        config['num_aps'] = int(args.num_aps_override)
    if (args.backhaul_lo_override is None) != (args.backhaul_hi_override is None):
        parser.error('pass both --backhaul_lo_override and --backhaul_hi_override')
    if args.backhaul_lo_override is not None:
        if float(args.backhaul_lo_override) < 0.0 or float(args.backhaul_hi_override) < float(args.backhaul_lo_override):
            parser.error('Require 0 <= backhaul lower bound <= upper bound')
        config['backhaul_lo'] = float(args.backhaul_lo_override)
        config['backhaul_hi'] = float(args.backhaul_hi_override)
    if args.cluster_policy is not None:
        config['cluster_policy'] = args.cluster_policy
    if args.baseline_policy == 'fixed' and (not args.fixed_action):
        parser.error('--baseline_policy fixed requires --fixed_action')
    if args.baseline_policy == 'rssi' and args.fixed_action:
        parser.error('--fixed_action cannot be combined with RSSI policy')
    all_args = Namespace(**config)
    seeds = [args.seed_start + index for index in range(args.episodes)]
    evaluation_range = seed_range(args.seed_start, args.episodes)
    selection = load_checkpoint_selection(run_dir)
    selection_range = None
    selection_overlap = False
    if selection is not None:
        selection_range = seed_range(int(selection['selection_seed_start']), int(selection['selection_episodes']))
        selection_overlap = ranges_overlap(evaluation_range, selection_range)
    if args.phase == 'test' and selection_overlap and (not args.allow_seed_overlap):
        parser.error(f'test seeds overlap checkpoint-selection seeds: test={evaluation_range[0]}..{evaluation_range[1]}, selection={selection_range[0]}..{selection_range[1]}. Choose disjoint seeds or explicitly pass --allow_seed_overlap for a diagnostic run.')
    env = IICMSMobileDiscreteActionEnv()
    env.configure(all_args)
    resolved_obs_dim = config.get('resolved_obs_dim')
    if resolved_obs_dim is not None and int(resolved_obs_dim) != int(env.signal_obs_dim):
        raise ValueError(f'AP-count override changed the policy observation dimension: checkpoint expects {resolved_obs_dim}, environment produces {env.signal_obs_dim}. This model cannot be evaluated zero-shot.')
    all_args.actor_token_dim = env.obs_token_dim
    all_args.critic_token_dim = env.signal_obs_dim
    all_args.joint_num_modes = env.joint_num_modes
    all_args.joint_num_sizes = env.joint_num_sizes
    all_args.joint_num_actions = env.joint_num_actions
    all_args.joint_valid_action_indices = list(env.joint_action_indices)
    all_args.joint_num_cluster_choices = int(env.env.joint_num_cluster_choices)
    all_args.joint_cluster_choice_ranks = [list(ranks) for ranks in env.env.cluster_choice_ranks]
    all_args.joint_cluster_choice_sizes = list(env.env.cluster_choice_sizes)
    all_args.joint_action_cluster_choice = list(env.env.joint_action_cluster_choice)
    all_args.joint_num_ap_tokens = int(env.joint_num_ap_tokens)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    separate_policy = bool(config.get('original_mappo', False)) or not bool(config.get('share_policy', True))
    policy = None
    policies = None
    fixed_action_name = None
    fixed_action_index = None
    actor_path = None
    actor_paths = None
    rssi_size = None
    rssi_actions = None
    effective_baseline_policy = 'fixed' if args.fixed_action else args.baseline_policy
    if args.fixed_action:
        (fixed_action_name, fixed_action_index) = resolve_fixed_action(env, args.fixed_action)
        execution_mode = 'fixed'
        policy_label = args.policy_label or fixed_action_name
    elif args.baseline_policy == 'rssi':
        rssi_size = int(args.rssi_cluster_size) if args.rssi_cluster_size is not None else int(max(env.env.size_options))
        rssi_actions = rssi_action_indices(env, rssi_size)
        execution_mode = f"rssi_{config['cluster_policy']}"
        policy_label = args.policy_label or f"rssi_{config['cluster_policy']}_l{rssi_size}"
    else:
        actor_paths = checkpoint_paths(run_dir, args.checkpoint, env.num_agent, separate_policy)
        actor_path = actor_paths[0]
        if separate_policy:
            policies = []
            for (agent_id, path) in enumerate(actor_paths):
                agent_policy = RMAPPOPolicy(all_args, env.observation_space[agent_id], env.share_observation_space[agent_id], env.action_space[agent_id], device=device)
                agent_policy.actor.load_state_dict(torch.load(path, map_location=device))
                critic_path = run_dir / 'models' / f'critic_agent{agent_id}.pt'
                if critic_path.exists():
                    agent_policy.critic.load_state_dict(torch.load(critic_path, map_location=device))
                agent_policy.actor.eval()
                agent_policy.critic.eval()
                policies.append(agent_policy)
        else:
            policy = RMAPPOPolicy(all_args, env.observation_space[0], env.share_observation_space[0], env.action_space[0], device=device)
            policy.actor.load_state_dict(torch.load(actor_path, map_location=device))
            critic_path = run_dir / 'models' / 'critic.pt'
            if critic_path.exists():
                policy.critic.load_state_dict(torch.load(critic_path, map_location=device))
            policy.actor.eval()
            policy.critic.eval()
        execution_mode = 'stochastic' if args.stochastic else 'deterministic'
        if separate_policy:
            policy_label = args.policy_label or 'original_mappo'
        else:
            policy_label = args.policy_label or f"{getattr(all_args, 'actor_backbone', 'mlp')}_{args.checkpoint}"
    (step_sum_rates, per_ue_rates, step_sum_se) = ([], [], [])
    mode_counts = np.zeros(3, dtype=float)
    size_counts = {str(size): 0.0 for size in env.env.size_options}
    joint_action_counts = {name: 0.0 for name in env.joint_action_names}
    (overload, backhaul_scale, cluster_change, rho2) = ([], [], [], [])
    (fronthaul_queue, fronthaul_drop, fronthaul_delay) = ([], [], [])
    (qos_violation, qos_shortfall) = ([], [])
    mean_ue_speed = []
    scene_digests = []
    total_decisions = 0
    evaluation_started = time.monotonic()
    for episode in range(args.episodes):
        seed = seeds[episode]
        episode_step_start = len(step_sum_rates)
        episode_args = Namespace(**config)
        episode_args.seed = seed
        env.configure(episode_args)
        torch.manual_seed(seed)
        obs = env.reset()
        scene_digests.append(initial_state_digest(env))
        rnn = np.zeros((env.num_agent, 1, 1), dtype=np.float32)
        masks = np.ones((env.num_agent, 1), dtype=np.float32)
        for _ in range(args.horizon):
            if policy is None and policies is None:
                if args.baseline_policy == 'rssi':
                    modes = rssi_modes(env, args.rssi_threshold_hi, args.rssi_threshold_lo)
                    env_actions = [encode_action(env, np.asarray([rssi_actions[int(mode)]])) for mode in modes]
                else:
                    env_actions = [encode_action(env, np.asarray([fixed_action_index])) for _ in range(env.num_agent)]
            elif policies is not None:
                action_rows = []
                for (agent_id, agent_policy) in enumerate(policies):
                    with torch.no_grad():
                        (action, rnn[agent_id]) = agent_policy.act(torch.as_tensor(obs[agent_id:agent_id + 1], device=device), torch.as_tensor(rnn[agent_id:agent_id + 1], device=device), torch.as_tensor(masks[agent_id:agent_id + 1], device=device), deterministic=not args.stochastic)
                    action_rows.append(action.detach().cpu().numpy())
                action = np.concatenate(action_rows, axis=0)
                env_actions = [encode_action(env, row) for row in action]
            else:
                with torch.no_grad():
                    (action, rnn) = policy.act(torch.as_tensor(obs, device=device), torch.as_tensor(rnn, device=device), torch.as_tensor(masks, device=device), deterministic=not args.stochastic)
                action = action.detach().cpu().numpy().astype(int)
                env_actions = [encode_action(env, row) for row in action]
            (obs, _, _, infos) = env.step(env_actions)
            rates = np.asarray([item['inst_rate'] for item in infos]) / 1000000.0
            sum_rate = float(rates.sum())
            system_se = sum_rate * 1000000.0 / float(env.env.bandwidth_hz)
            step_sum_rates.append(sum_rate)
            step_sum_se.append(system_se)
            per_ue_rates.extend(rates.tolist())
            overload.append(float(np.mean([item['backhaul_scale'] < 0.999999 for item in infos])))
            backhaul_scale.append(float(np.mean([item['backhaul_scale'] for item in infos])))
            fronthaul_queue.append(float(np.mean([item['fronthaul_queue_ratio'] for item in infos])))
            fronthaul_drop.append(float(np.mean([item['fronthaul_drop_ratio'] for item in infos])))
            fronthaul_delay.append(float(np.mean([item['fronthaul_queue_delay_ms'] for item in infos])))
            cluster_change.append(float(np.mean([item['cluster_changed'] for item in infos])))
            rho2.append(float(np.mean([item['rho2'] for item in infos])))
            qos_violation.append(float(np.mean([item['qos_violation'] for item in infos])))
            qos_shortfall.append(float(np.mean([item['qos_shortfall_mbps'] for item in infos])))
            mean_ue_speed.append(float(np.mean([item['speed'] for item in infos])))
            for item in infos:
                mode_counts[int(item['mode'])] += 1.0
                joint_action_counts[item['joint_action_name']] += 1.0
                if int(item['mode']) != 2:
                    size_counts[str(int(item['cluster_k']))] += 1.0
            total_decisions += env.num_agent
        episode_sum_rates = np.asarray(step_sum_rates[episode_step_start:], dtype=float)
        print(
            f"[evaluate_iicms] episode {episode + 1}/{args.episodes} "
            f"seed={seed} "
            f"mean_sum_rate={float(np.mean(episode_sum_rates)):.2f} Mbps "
            f"elapsed={time.monotonic() - evaluation_started:.1f}s",
            flush=True,
        )
    mode_share = mode_counts / max(1, total_decisions)
    size_total = max(1.0, sum(size_counts.values()))
    per_ue_se_samples = np.asarray(per_ue_rates, dtype=float) * 1000000.0 / float(env.env.bandwidth_hz)
    (rate_grid, rate_cdf) = empirical_cdf(per_ue_rates)
    (se_grid, se_cdf) = empirical_cdf(per_ue_se_samples)
    cdf_filename = f'{run_dir.name}_rate_cdf.npz'
    rate_requirement_se = float(env.env.rate_req_lo_mbps) * 1000000.0 / float(env.env.bandwidth_hz)
    min_ue_se_floor = float(getattr(env.env, 'min_ue_se_bps_hz', 0.0))
    min_ue_rate_floor = min_ue_se_floor * float(env.env.bandwidth_hz) / 1000000.0
    env_protocol = canonical_env_config(config)
    evaluation_protocol = {'phase': args.phase, 'seed_start': args.seed_start, 'episodes': args.episodes, 'seed_range': list(evaluation_range), 'seeds': seeds, 'horizon': args.horizon, 'execution_mode': execution_mode, 'scenario_seed_policy': 'evaluation_episode_seed', 'initial_state_digests': scene_digests, 'environment_config': env_protocol, 'test_protocol_id': protocol_id(config, seeds, args.horizon, args.phase, scene_digests)}
    checkpoint_info = None
    if selection is not None:
        checkpoint_info = {'selection_seed_start': selection['selection_seed_start'], 'selection_episodes': selection['selection_episodes'], 'selection_horizon': selection.get('selection_horizon'), 'selection_metric': selection.get('selection_metric'), 'selected_step': selection.get('selected_step'), 'evaluation_overlaps_selection': selection_overlap}
    independent_test = args.phase == 'test' and (not selection_overlap) and (selection is not None)
    is_learned_policy = policy is not None or policies is not None
    result = {'run_dir': str(run_dir), 'config_source_run_dir': str(run_dir), 'policy_label': policy_label, 'policy_type': 'rssi_threshold' if args.baseline_policy == 'rssi' else 'learned' if is_learned_policy else 'fixed_action', 'checkpoint_role': args.checkpoint if is_learned_policy else None, 'checkpoint_file': None if actor_path is None else actor_path.name, 'checkpoint_files': None if actor_paths is None else [path.name for path in actor_paths], 'fixed_action': fixed_action_name, 'fixed_action_index': fixed_action_index, 'baseline_policy': effective_baseline_policy, 'cluster_policy': config['cluster_policy'], 'num_aps_override': args.num_aps_override, 'backhaul_lo_override': args.backhaul_lo_override, 'backhaul_hi_override': args.backhaul_hi_override, 'rssi_cluster_size': rssi_size, 'rssi_threshold_hi': args.rssi_threshold_hi if args.baseline_policy == 'rssi' else None, 'rssi_threshold_lo': args.rssi_threshold_lo if args.baseline_policy == 'rssi' else None, 'actor_backbone': ('rssi' if args.baseline_policy == 'rssi' else 'fixed') if not is_learned_policy else getattr(all_args, 'actor_backbone', 'mlp'), 'critic_backbone': 'none' if not is_learned_policy else getattr(all_args, 'critic_backbone', 'mlp'), 'episodes': args.episodes, 'horizon': args.horizon, 'objective_horizon': int(env.env.objective_horizon), 'objective_seconds': args.horizon * float(env.env.slot_dt), 'evaluation_seed_start': args.seed_start, 'deterministic': execution_mode != 'stochastic', 'execution_mode': execution_mode, 'evaluation_phase': args.phase, 'independent_test': independent_test, 'checkpoint_selection': checkpoint_info, 'evaluation_protocol': evaluation_protocol, 'evaluation_seeds': seeds, 'time_averaged_system_se_bps_hz': float(np.mean(step_sum_se)), 'time_averaged_mean_ue_se_bps_hz': float(np.mean(step_sum_se) / env.num_agent), 'system_se_std_bps_hz': float(np.std(step_sum_se)), 'sum_rate_mbps': float(np.mean(step_sum_rates)), 'sum_rate_std_mbps': float(np.std(step_sum_rates)), 'per_ue_rate_p05_mbps': float(np.percentile(per_ue_rates, 5)), 'per_ue_rate_p50_mbps': float(np.percentile(per_ue_rates, 50)), 'per_ue_rate_p95_mbps': float(np.percentile(per_ue_rates, 95)), 'rate_requirement_se_bps_hz': rate_requirement_se, 'min_ue_se_bps_hz': min_ue_se_floor, 'min_ue_rate_mbps': min_ue_rate_floor, 'minimum_observed_ue_se_bps_hz': float(np.min(per_ue_se_samples)), 'se_floor_violation_rate': float(np.mean(per_ue_se_samples < min_ue_se_floor)) if min_ue_se_floor > 0.0 else 0.0, 'per_ue_se_cdf_file': cdf_filename, 'per_ue_se_cdf': {'samples': int(per_ue_se_samples.size)}, 'overload_ratio': float(np.mean(overload)), 'backhaul_scale': float(np.mean(backhaul_scale)), 'fronthaul_queue_ratio': float(np.mean(fronthaul_queue)), 'fronthaul_drop_ratio': float(np.mean(fronthaul_drop)), 'fronthaul_queue_delay_ms': float(np.mean(fronthaul_delay)), 'min_rate_requirement_mbps': float(env.env.rate_req_lo_mbps), 'qos_violation_rate': float(np.mean(qos_violation)), 'qos_shortfall_mbps': float(np.mean(qos_shortfall)), 'mean_ue_speed_mps': float(np.mean(mean_ue_speed)), 'cluster_change_rate': float(np.mean(cluster_change)), 'rho2': float(np.mean(rho2)), 'mode_share': {'JT': float(mode_share[0]), 'CB': float(mode_share[1]), 'NC': float(mode_share[2])}, 'cluster_size_share_cb_jt': {size: count / size_total for (size, count) in size_counts.items()}, 'joint_action_share': {name: count / max(1, total_decisions) for (name, count) in joint_action_counts.items()}, 'config': config, 'source_sha256': {name: hashlib.sha256((Path(__file__).resolve().parent / name).read_bytes()).hexdigest() for name in ('envs/env_core.py', 'envs/env_iicms_mobile.py', 'evaluate_iicms.py', 'algorithms/algorithm/r_actor_critic.py', 'algorithms/utils/act.py')}}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / cdf_filename, rate_mbps=rate_grid, rate_cdf=rate_cdf, se_bps_hz=se_grid, cdf=se_cdf, samples=np.asarray([per_ue_se_samples.size], dtype=np.int64))
    path = out / f'{run_dir.name}.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
