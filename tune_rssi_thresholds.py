from __future__ import annotations
import argparse
import csv
import json
import re
from argparse import Namespace
from pathlib import Path
from utils.runtime_env import normalize_thread_environment
normalize_thread_environment()
import numpy as np
from envs.env_iicms_mobile import IICMSMobileDiscreteActionEnv
from evaluate_iicms import encode_action, rssi_action_indices, rssi_modes

def parse_grid(value: str) -> list[float]:
    values = sorted({float(item.strip()) for item in value.split(',') if item.strip()})
    if not values or values[0] <= 0.0:
        raise ValueError('Threshold grid must contain positive values')
    return values

def resolve_run_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / 'config.json').is_file():
        return path
    runs = []
    for config_path in path.glob('run*/config.json'):
        match = re.fullmatch('run(\\d+)', config_path.parent.name)
        if match:
            runs.append((int(match.group(1)), config_path.parent))
    if runs:
        return max(runs)[1]
    raise FileNotFoundError(f'No config.json found in --run_dir or its runN subdirectories: {path}')

def evaluate_policy(config: dict, cluster_policy: str, low: float, high: float, seeds: list[int], horizon: int, cluster_size: int) -> dict:
    cfg = dict(config)
    cfg['cluster_policy'] = cluster_policy
    env = IICMSMobileDiscreteActionEnv()
    env.configure(Namespace(**cfg))
    action_indices = rssi_action_indices(env, cluster_size)
    episode_rates = []
    per_step_rates = []
    qos = []
    overload = []
    queue_delay = []
    mode_counts = np.zeros(3, dtype=np.int64)
    for seed in seeds:
        episode_args = Namespace(**cfg)
        episode_args.seed = seed
        env.configure(episode_args)
        env.reset()
        current_episode_rates = []
        for _ in range(horizon):
            modes = rssi_modes(env, high, low)
            actions = [encode_action(env, np.asarray([action_indices[int(mode)]])) for mode in modes]
            (_, _, _, infos) = env.step(actions)
            rates = np.asarray([item['inst_rate'] for item in infos], dtype=float) / 1000000.0
            rate_sum = float(np.sum(rates))
            current_episode_rates.append(rate_sum)
            per_step_rates.append(rate_sum)
            qos.append(float(np.mean([item['qos_violation'] for item in infos])))
            overload.append(float(np.mean([item['backhaul_scale'] < 0.999999 for item in infos])))
            queue_delay.append(float(np.mean([item['fronthaul_queue_delay_ms'] for item in infos])))
            mode_counts += np.bincount(modes, minlength=3)
        episode_rates.append(float(np.mean(current_episode_rates)))
    total_modes = max(1, int(mode_counts.sum()))
    return {'cluster_policy': cluster_policy, 'rssi_threshold_lo': float(low), 'rssi_threshold_hi': float(high), 'validation_seeds': seeds, 'horizon': int(horizon), 'sum_rate_mbps': float(np.mean(per_step_rates)), 'episode_rate_sd_mbps': float(np.std(episode_rates)), 'p05_step_rate_mbps': float(np.percentile(per_step_rates, 5)), 'qos_violation_rate': float(np.mean(qos)), 'overload_ratio': float(np.mean(overload)), 'fronthaul_queue_delay_ms': float(np.mean(queue_delay)), 'mode_share': {'JT': float(mode_counts[0] / total_modes), 'CB': float(mode_counts[1] / total_modes), 'NC': float(mode_counts[2] / total_modes)}}

def evaluate_pair(config: dict, low: float, high: float, seeds: list[int], horizon: int, cluster_size: int) -> dict:
    static = evaluate_policy(config, 'static', low, high, seeds, horizon, cluster_size)
    dynamic = evaluate_policy(config, 'dynamic', low, high, seeds, horizon, cluster_size)
    return {'rssi_threshold_lo': float(low), 'rssi_threshold_hi': float(high), 'shared_mean_rate_mbps': 0.5 * (static['sum_rate_mbps'] + dynamic['sum_rate_mbps']), 'static': static, 'dynamic': dynamic}

def rank_key(result: dict) -> tuple[float, float, float]:
    return (float(result['shared_mean_rate_mbps']), -float(result['static']['qos_violation_rate']), -float(result['dynamic']['qos_violation_rate']))

def boundary_warnings(selected: dict, low_grid: list[float], high_grid: list[float]) -> list[str]:
    entries = [('shared', selected['shared']), ('static', selected['per_policy']['static']), ('dynamic', selected['per_policy']['dynamic'])]
    warnings = []
    for (label, item) in entries:
        if np.isclose(item['rssi_threshold_lo'], min(low_grid)):
            warnings.append(f"{label} low threshold {item['rssi_threshold_lo']} is at the lower grid boundary")
        if np.isclose(item['rssi_threshold_hi'], max(high_grid)):
            warnings.append(f"{label} high threshold {item['rssi_threshold_hi']} is at the upper grid boundary")
    return warnings

def write_outputs(out_dir: Path, result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'rssi_threshold_calibration.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = []
    for phase in ('screening', 'final'):
        for item in result[phase]:
            for policy in ('static', 'dynamic'):
                rows.append({'phase': phase, 'rssi_threshold_lo': item['rssi_threshold_lo'], 'rssi_threshold_hi': item['rssi_threshold_hi'], 'cluster_policy': policy, **item[policy]})
    fieldnames = sorted({key for row in rows for key in row})
    ordered = ['phase', 'cluster_policy', 'rssi_threshold_lo', 'rssi_threshold_hi', *[key for key in fieldnames if key not in {'phase', 'cluster_policy', 'rssi_threshold_lo', 'rssi_threshold_hi'}]]
    with (out_dir / 'rssi_threshold_calibration.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row['phase']))
    selected = result['selected']
    shared = selected['shared']
    per_policy = selected['per_policy']
    env_text = f"RSSI_THRESHOLD_LO={shared['rssi_threshold_lo']}\nRSSI_THRESHOLD_HI={shared['rssi_threshold_hi']}\nSC_RSSI_THRESHOLD_LO={per_policy['static']['rssi_threshold_lo']}\nSC_RSSI_THRESHOLD_HI={per_policy['static']['rssi_threshold_hi']}\nDC_RSSI_THRESHOLD_LO={per_policy['dynamic']['rssi_threshold_lo']}\nDC_RSSI_THRESHOLD_HI={per_policy['dynamic']['rssi_threshold_hi']}\n"
    (out_dir / 'selected_rssi_thresholds.env').write_text(env_text, encoding='utf-8')
    print(json.dumps(selected, ensure_ascii=False, indent=2))
    for warning in result.get('boundary_warnings', []):
        print(f'[WARN] {warning}', flush=True)
    print(f'OUTPUT_DIR={out_dir}')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--validation_seed_start', type=int, default=92000)
    parser.add_argument('--screen_episodes', type=int, default=5)
    parser.add_argument('--final_episodes', type=int, default=20)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--cluster_size', type=int, default=5)
    parser.add_argument('--low_grid', default='0.05,0.1,0.25,0.5,1,2,4')
    parser.add_argument('--high_grid', default='0.5,1,2,4,8,16,32,64,128,256')
    parser.add_argument('--finalists', type=int, default=5)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    if args.screen_episodes < 1 or args.final_episodes < 1:
        parser.error('episode counts must be positive')
    if args.finalists < 1:
        parser.error('--finalists must be positive')
    if args.horizon < 1:
        parser.error('--horizon must be positive')
    try:
        run_dir = resolve_run_dir(Path(args.run_dir))
    except FileNotFoundError as error:
        parser.error(str(error))
    print(f'[calibration] run_dir={run_dir}', flush=True)
    config = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    low_grid = parse_grid(args.low_grid)
    high_grid = parse_grid(args.high_grid)
    pairs = [(low, high) for low in low_grid for high in high_grid if low < high]
    for required in ((1.0, 20.0),):
        if required not in pairs:
            pairs.append(required)
    screen_seeds = [args.validation_seed_start + index for index in range(args.screen_episodes)]
    screening = [evaluate_pair(config, low, high, screen_seeds, args.horizon, args.cluster_size) for (low, high) in pairs]
    finalist_pairs = [(item['rssi_threshold_lo'], item['rssi_threshold_hi']) for item in sorted(screening, key=rank_key, reverse=True)[:args.finalists]]
    final_seeds = [args.validation_seed_start + index for index in range(args.final_episodes)]
    final = [evaluate_pair(config, low, high, final_seeds, args.horizon, args.cluster_size) for (low, high) in finalist_pairs]
    selected_shared = max(final, key=rank_key)
    selected_static = max(final, key=lambda item: (item['static']['sum_rate_mbps'], -item['static']['qos_violation_rate']))
    selected_dynamic = max(final, key=lambda item: (item['dynamic']['sum_rate_mbps'], -item['dynamic']['qos_violation_rate']))
    result = {'selection_objective': 'maximize mean of SC-RSSI and DC-RSSI system rate; use lower QoS violation as tie-break', 'validation_seed_range': [args.validation_seed_start, args.validation_seed_start + args.final_episodes - 1], 'screen_episodes': args.screen_episodes, 'final_episodes': args.final_episodes, 'horizon': args.horizon, 'rssi_cluster_size': args.cluster_size, 'candidate_pairs': [{'lo': low, 'hi': high} for (low, high) in pairs], 'screening': screening, 'final': final, 'selected': {'shared': {'rssi_threshold_lo': selected_shared['rssi_threshold_lo'], 'rssi_threshold_hi': selected_shared['rssi_threshold_hi'], 'shared_mean_rate_mbps': selected_shared['shared_mean_rate_mbps'], 'static': selected_shared['static'], 'dynamic': selected_shared['dynamic']}, 'per_policy': {'static': selected_static['static'], 'dynamic': selected_dynamic['dynamic']}}}
    result['boundary_warnings'] = boundary_warnings(result['selected'], low_grid, high_grid)
    out_dir = Path(args.out) if args.out else run_dir / 'rssi_threshold_calibration'
    write_outputs(out_dir, result)
if __name__ == '__main__':
    main()
