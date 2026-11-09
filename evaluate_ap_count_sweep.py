#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
import math
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
FAMILY_LABELS = {'mlp': 'MLP-factorized (new)', 'old_mlp': 'MLP-flat (old)', 'tf': 'TF-native'}
BASELINE_LABELS = {'dc_rssi': 'DC-RSSI', 'sc_rssi': 'SC-RSSI', 'dc_cb_l5': 'DC-CB-L5', 'sc_cb_l5': 'SC-CB-L5', 'dc_jt_l2': 'DC-JT-L2', 'sc_jt_l3': 'SC-JT-L3', 'dc_nc': 'DC-NC', 'sc_nc': 'SC-NC'}
DEFAULT_BASELINES = ','.join(BASELINE_LABELS)

@dataclass(frozen=True)
class BaselineSpec:
    label: str
    cluster_policy: str
    fixed_action: str | None
    rssi: bool = False
BASELINE_SPECS = {'dc_rssi': BaselineSpec('dc_rssi', 'dynamic', None, True), 'sc_rssi': BaselineSpec('sc_rssi', 'static', None, True), 'dc_cb_l5': BaselineSpec('dc_cb_l5', 'dynamic', 'CB_L5_S00'), 'sc_cb_l5': BaselineSpec('sc_cb_l5', 'static', 'CB_L5_S00'), 'dc_jt_l2': BaselineSpec('dc_jt_l2', 'dynamic', 'JT_L2_S00'), 'sc_jt_l3': BaselineSpec('sc_jt_l3', 'static', 'JT_L3_S00'), 'dc_nc': BaselineSpec('dc_nc', 'dynamic', 'NC'), 'sc_nc': BaselineSpec('sc_nc', 'static', 'NC')}

def parse_int_list(value: str, name: str) -> list[int]:
    values = []
    for item in str(value).split(','):
        item = item.strip()
        if item:
            values.append(int(item))
    values = sorted(set(values))
    if not values:
        raise ValueError(f'{name} must not be empty')
    return values

def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(',') if item.strip()]

def run_dir_for_family(root: Path, tag: str, family: str, seed: int) -> Path:
    suffix = 'mlp' if family in {'mlp', 'old_mlp'} else 'tf_actor'
    return root / f'{tag}_{suffix}_s{seed}' / 'run1'

def find_result_json(directory: Path) -> Path:
    candidates = sorted(directory.glob('*.json'))
    if not candidates:
        raise FileNotFoundError(f'No evaluation JSON under {directory}')
    return candidates[-1]

def validate_reusable_result(path: Path, seed_start: int, episodes: int, horizon: int) -> bool:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return False
    return int(data.get('evaluation_seed_start', -1)) == seed_start and int(data.get('episodes', -1)) == episodes and (int(data.get('horizon', -1)) == horizon) and bool(data.get('deterministic', False))

def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        (key, value) = line.split('=', 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values

def candidate_threshold_file(requested: str, template_run: Path) -> Path | None:
    if requested:
        path = Path(requested)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    candidate = template_run / 'rssi_threshold_calibration' / 'selected_rssi_thresholds.env'
    return candidate if candidate.exists() else None

def execute_evaluation(command: list[str], cwd: Path, log_path: Path, dry_run: bool) -> None:
    print('RUN', shlex.join(command), flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log_path.write_text(completed.stdout, encoding='utf-8')
    if completed.returncode != 0:
        tail = '\n'.join(completed.stdout.splitlines()[-30:])
        raise RuntimeError(f'Evaluation failed ({completed.returncode}). See {log_path}\n{tail}')

def add_common_eval_args(command: list[str], run_dir: Path, output_dir: Path, ap_count: int, seed_start: int, episodes: int, horizon: int) -> None:
    command.extend(['--run_dir', str(run_dir), '--seed_start', str(seed_start), '--episodes', str(episodes), '--horizon', str(horizon), '--phase', 'test', '--num_aps_override', str(ap_count), '--out', str(output_dir)])

def confidence_radius(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    critical = 1.96
    try:
        from scipy.stats import t as student_t
        critical = float(student_t.ppf(0.975, df=len(values) - 1))
    except Exception:
        critical = 2.571 if len(values) == 6 else 1.96
    return critical * statistics.stdev(values) / math.sqrt(len(values))

def aggregate_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        groups.setdefault((row['strategy'], int(row['ap_count'])), []).append(row)
    metrics = ('sum_rate_mbps', 'rate_per_ap_mbps', 'per_ue_rate_p05_mbps', 'qos_violation_rate', 'overload_ratio', 'fronthaul_queue_delay_ms', 'cluster_change_rate')
    aggregated = []
    for ((strategy, ap_count), group) in sorted(groups.items()):
        output = {'strategy': strategy, 'strategy_label': group[0]['strategy_label'], 'ap_count': ap_count, 'model_seed_count': len({item['model_seed'] for item in group if item['model_seed'] is not None})}
        for metric in metrics:
            values = [float(item[metric]) for item in group if item.get(metric) is not None]
            output[f'{metric}_mean'] = statistics.mean(values) if values else None
            output[f'{metric}_ci95'] = confidence_radius(values) if values else None
        aggregated.append(output)
    return aggregated

def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields = sorted({key for row in rows for key in row})
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def plot_sweep(aggregate: list[dict], ap_counts: list[int], output_path: Path) -> None:
    strategies = sorted({row['strategy'] for row in aggregate}, key=lambda value: (0 if value == 'mlp' else 1 if value == 'old_mlp' else 2 if value == 'tf' else 3, value))
    colors = plt.get_cmap('tab10').colors
    markers = ('o', 's', '^', 'v', 'D', 'P', 'X', '*', '<', '>')
    styles = {strategy: (colors[index % len(colors)], markers[index % len(markers)]) for (index, strategy) in enumerate(strategies)}
    panels = (('sum_rate_mbps', 'Total sum rate (Mbps)', 'Total system rate'), ('rate_per_ap_mbps', 'Sum rate per AP (Mbps/AP)', 'Average rate per AP'), ('per_ue_rate_p05_mbps', 'Per-UE rate P05 (Mbps)', 'Low-rate tail'), ('qos_violation_rate', 'QoS violation rate', '40 Mbps QoS violation'))
    (fig, axes) = plt.subplots(2, 2, figsize=(12.5, 8.2), dpi=180)
    for (axis, (metric, ylabel, title)) in zip(axes.ravel(), panels):
        for strategy in strategies:
            selected = sorted((row for row in aggregate if row['strategy'] == strategy), key=lambda row: row['ap_count'])
            if not selected:
                continue
            x = np.asarray([row['ap_count'] for row in selected], dtype=float)
            mean = np.asarray([row[f'{metric}_mean'] for row in selected], dtype=float)
            ci = np.asarray([row[f'{metric}_ci95'] or 0.0 for row in selected], dtype=float)
            (color, marker) = styles[strategy]
            if np.any(ci > 0.0):
                axis.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.1, linewidth=0)
            axis.plot(x, mean, color=color, marker=marker, linewidth=2.0, markersize=5, label=selected[0]['strategy_label'])
        axis.axvline(30, color='#777777', linestyle=':', linewidth=1.0)
        axis.set_xticks(ap_counts)
        axis.set_xlabel('Number of APs')
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, alpha=0.22, linewidth=0.8)
    (handles, labels) = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, frameon=False, fontsize=8.5)
    fig.suptitle('Zero-shot AP-count scalability (policies trained with 30 APs)', fontsize=14)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate fixed M=30 policy checkpoints at several AP counts without retraining.')
    parser.add_argument('--result-root', default='results/IICMSMobility/PureMobility/mappo')
    parser.add_argument('--mlp-tag', default='fronthaul_queue_mlp_factorized_rate_400k')
    parser.add_argument('--tf-tag', default='fronthaul_queue_native_rate_400k')
    parser.add_argument('--old-mlp-tag', default='fronthaul_queue_native_rate_400k')
    parser.add_argument('--mlp-seeds', default='1,2,3,4,5,6')
    parser.add_argument('--tf-seeds', default='1,2,3,4,5,6')
    parser.add_argument('--old-mlp-seeds', default='1,2,3,4,5,6')
    parser.add_argument('--families', default='mlp,old_mlp,tf', help='learned families: mlp, old_mlp, tf')
    parser.add_argument('--ap-counts', default='10,20,30,40,50')
    parser.add_argument('--baselines', default=DEFAULT_BASELINES, help="comma-separated baselines or 'none'")
    parser.add_argument('--rssi-thresholds-env', default='', help='selected_rssi_thresholds.env; defaults to the calibration file under the first MLP run')
    parser.add_argument('--rssi-cluster-size', type=int, default=5)
    parser.add_argument('--seed-start', type=int, default=93000)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--output-dir', default='analysis_outputs/ap_count_zero_shot')
    parser.add_argument('--python-bin', default=sys.executable)
    parser.add_argument('--reuse-m30', action='store_true', default=True, help='reuse existing deterministic M=30 evaluation JSON when it matches the requested seeds, episodes, and horizon')
    parser.add_argument('--no-reuse-m30', dest='reuse_m30', action='store_false')
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent
    result_root = (project_root / args.result_root).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ap_counts = parse_int_list(args.ap_counts, 'AP counts')
    families = parse_str_list(args.families)
    unknown_families = set(families) - {'mlp', 'old_mlp', 'tf'}
    if unknown_families:
        parser.error(f'Unknown families: {sorted(unknown_families)}')
    baselines = [] if args.baselines.strip().lower() == 'none' else parse_str_list(args.baselines)
    unknown_baselines = set(baselines) - set(BASELINE_SPECS)
    if unknown_baselines:
        parser.error(f'Unknown baselines: {sorted(unknown_baselines)}')
    seed_lists = {'mlp': parse_int_list(args.mlp_seeds, 'MLP seeds'), 'old_mlp': parse_int_list(args.old_mlp_seeds, 'old MLP seeds'), 'tf': parse_int_list(args.tf_seeds, 'TF seeds')}
    tags = {'mlp': args.mlp_tag, 'old_mlp': args.old_mlp_tag, 'tf': args.tf_tag}
    family_runs = {family: {seed: run_dir_for_family(result_root, tags[family], family, seed) for seed in seed_lists[family]} for family in families}
    for (family, runs) in family_runs.items():
        for (seed, run_dir) in runs.items():
            if not (run_dir / 'config.json').exists():
                raise FileNotFoundError(f'Missing {family} seed {seed} run: {run_dir}')
    template_run = next((run for family in families for run in family_runs[family].values()))
    threshold_path = None
    if any((name.endswith('rssi') for name in baselines)):
        if args.rssi_thresholds_env:
            threshold_path = candidate_threshold_file(args.rssi_thresholds_env, template_run)
        else:
            for family in families:
                for run_dir in family_runs[family].values():
                    candidate = candidate_threshold_file('', run_dir)
                    if candidate is not None:
                        threshold_path = candidate
                        break
                if threshold_path is not None:
                    break
        if threshold_path is None:
            parser.error('RSSI baselines require --rssi-thresholds-env because no calibration file was found under the template run')
    thresholds = read_env_file(threshold_path) if threshold_path is not None else {}
    raw_rows: list[dict] = []
    result_paths: list[Path] = []
    for ap_count in ap_counts:
        print(f'\n===== AP count {ap_count} =====', flush=True)
        ap_dir = output_dir / f'ap_{ap_count:03d}'
        for family in families:
            display = FAMILY_LABELS[family]
            for (seed, run_dir) in family_runs[family].items():
                eval_dir = ap_dir / f'{family}_s{seed}'
                existing_json = None
                if ap_count == 30 and args.reuse_m30 and (run_dir / 'evaluation_deterministic').is_dir():
                    candidates = sorted((run_dir / 'evaluation_deterministic').glob('*.json'))
                    for candidate in reversed(candidates):
                        if validate_reusable_result(candidate, args.seed_start, args.episodes, args.horizon):
                            existing_json = candidate
                            break
                if existing_json is None:
                    command = [args.python_bin, 'evaluate_iicms.py']
                    add_common_eval_args(command, run_dir, eval_dir, ap_count, args.seed_start, args.episodes, args.horizon)
                    command.extend(['--checkpoint', 'actor'])
                    log_path = eval_dir / 'evaluate.log'
                    existing_candidates = sorted(eval_dir.glob('*.json')) if eval_dir.exists() else []
                    if args.skip_existing and existing_candidates:
                        result_path = existing_candidates[-1]
                    else:
                        execute_evaluation(command, project_root, log_path, args.dry_run)
                        if args.dry_run:
                            continue
                        result_path = find_result_json(eval_dir)
                else:
                    result_path = existing_json
                payload = json.loads(result_path.read_text(encoding='utf-8'))
                raw_rows.append({'strategy': family, 'strategy_label': display, 'ap_count': ap_count, 'model_seed': seed, 'sum_rate_mbps': payload['sum_rate_mbps'], 'rate_per_ap_mbps': float(payload['sum_rate_mbps']) / float(ap_count), 'per_ue_rate_p05_mbps': payload['per_ue_rate_p05_mbps'], 'qos_violation_rate': payload['qos_violation_rate'], 'overload_ratio': payload['overload_ratio'], 'fronthaul_queue_delay_ms': payload['fronthaul_queue_delay_ms'], 'cluster_change_rate': payload['cluster_change_rate'], 'mode_jt': payload['mode_share']['JT'], 'mode_cb': payload['mode_share']['CB'], 'mode_nc': payload['mode_share']['NC'], 'evaluation_json': str(result_path)})
                result_paths.append(result_path)
        for baseline in baselines:
            spec = BASELINE_SPECS[baseline]
            eval_dir = ap_dir / 'baselines' / baseline
            command = [args.python_bin, 'evaluate_iicms.py']
            add_common_eval_args(command, template_run, eval_dir, ap_count, args.seed_start, args.episodes, args.horizon)
            if spec.rssi:
                low_key = f"{baseline.split('_')[0].upper()}_RSSI_THRESHOLD_LO"
                high_key = f"{baseline.split('_')[0].upper()}_RSSI_THRESHOLD_HI"
                low = thresholds.get(low_key, thresholds.get('RSSI_THRESHOLD_LO'))
                high = thresholds.get(high_key, thresholds.get('RSSI_THRESHOLD_HI'))
                if low is None or high is None:
                    raise ValueError(f'Missing RSSI thresholds {low_key}/{high_key}')
                command.extend(['--baseline_policy', 'rssi', '--rssi_cluster_size', str(args.rssi_cluster_size), '--rssi_threshold_lo', str(low), '--rssi_threshold_hi', str(high), '--cluster_policy', spec.cluster_policy, '--policy_label', baseline])
            else:
                command.extend(['--baseline_policy', 'fixed', '--fixed_action', spec.fixed_action, '--cluster_policy', spec.cluster_policy, '--policy_label', baseline])
            log_path = eval_dir / 'evaluate.log'
            existing_candidates = sorted(eval_dir.glob('*.json')) if eval_dir.exists() else []
            if args.skip_existing and existing_candidates:
                result_path = existing_candidates[-1]
            else:
                execute_evaluation(command, project_root, log_path, args.dry_run)
                if args.dry_run:
                    continue
                result_path = find_result_json(eval_dir)
            payload = json.loads(result_path.read_text(encoding='utf-8'))
            raw_rows.append({'strategy': baseline, 'strategy_label': BASELINE_LABELS[baseline], 'ap_count': ap_count, 'model_seed': None, 'sum_rate_mbps': payload['sum_rate_mbps'], 'rate_per_ap_mbps': float(payload['sum_rate_mbps']) / float(ap_count), 'per_ue_rate_p05_mbps': payload['per_ue_rate_p05_mbps'], 'qos_violation_rate': payload['qos_violation_rate'], 'overload_ratio': payload['overload_ratio'], 'fronthaul_queue_delay_ms': payload['fronthaul_queue_delay_ms'], 'cluster_change_rate': payload['cluster_change_rate'], 'mode_jt': payload['mode_share']['JT'], 'mode_cb': payload['mode_share']['CB'], 'mode_nc': payload['mode_share']['NC'], 'evaluation_json': str(result_path)})
            result_paths.append(result_path)
    if args.dry_run:
        print('\nDRY RUN: no summary or plots generated.')
        return
    aggregated = aggregate_rows(raw_rows)
    write_csv(output_dir / 'raw_results.csv', raw_rows)
    write_csv(output_dir / 'aggregate_results.csv', aggregated)
    digest_groups: dict[int, set[tuple[str, ...]]] = {}
    for path in result_paths:
        payload = json.loads(path.read_text(encoding='utf-8'))
        ap_count = int(payload.get('num_aps_override') or payload['evaluation_protocol']['environment_config']['num_aps'])
        digests = tuple(payload['evaluation_protocol']['initial_state_digests'])
        digest_groups.setdefault(ap_count, set()).add(digests)
    manifest = {'analysis': 'zero-shot AP-count scalability', 'trained_ap_count': 30, 'ap_counts': ap_counts, 'families': families, 'baselines': baselines, 'seed_start': args.seed_start, 'episodes': args.episodes, 'horizon': args.horizon, 'rssi_thresholds_env': str(threshold_path) if threshold_path else None, 'rssi_cluster_size': args.rssi_cluster_size, 'scenario_digests_consistent': {str(ap_count): len(groups) == 1 for (ap_count, groups) in sorted(digest_groups.items())}, 'raw_result_rows': len(raw_rows)}
    (output_dir / 'sweep_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    plot_sweep(aggregated, ap_counts, output_dir / 'ap_count_scalability.png')
    from plot_ap_count_results import render_improved_plots
    (main_figure, advantage_figure) = render_improved_plots(output_dir)
    print(f'OUTPUT_DIR={output_dir}')
    print(f"FIGURE={output_dir / 'ap_count_scalability.png'}")
    print(f'MAIN_FIGURE={main_figure}')
    print(f'ADVANTAGE_FIGURE={advantage_figure}')
if __name__ == '__main__':
    main()
