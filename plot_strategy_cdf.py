from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_seed_se_cdf import FAMILY_LABELS, latest_run_dir, load_cdf
POLICY_LABELS = {'dc_rssi': 'DC-RSSI', 'sc_rssi': 'SC-RSSI', 'dc_nc': 'DC-NC', 'sc_nc': 'SC-NC', 'dc_cb_l2': 'DC-CB-L2', 'dc_cb_l3': 'DC-CB-L3', 'dc_cb_l4': 'DC-CB-L4', 'dc_cb_l5': 'DC-CB-L5', 'sc_cb_l2': 'SC-CB-L2', 'sc_cb_l3': 'SC-CB-L3', 'sc_cb_l4': 'SC-CB-L4', 'sc_cb_l5': 'SC-CB-L5', 'dc_jt_l2': 'DC-JT-L2', 'dc_jt_l3': 'DC-JT-L3', 'dc_jt_l4': 'DC-JT-L4', 'dc_jt_l5': 'DC-JT-L5', 'sc_jt_l2': 'SC-JT-L2', 'sc_jt_l3': 'SC-JT-L3', 'sc_jt_l4': 'SC-JT-L4', 'sc_jt_l5': 'SC-JT-L5'}

def fixed_mode(policy: str) -> str | None:
    if policy in {'sc_nc', 'dc_nc'}:
        return 'NC'
    if policy.startswith(('sc_cb_l', 'dc_cb_l')):
        return 'CB'
    if policy.startswith(('sc_jt_l', 'dc_jt_l')):
        return 'JT'
    return None

def load_baselines(suite_dir: Path) -> dict[str, dict]:
    baselines = {}
    for result_path in sorted(suite_dir.glob('*/*.json')):
        payload = json.loads(result_path.read_text(encoding='utf-8'))
        policy = payload.get('policy_label')
        if policy not in POLICY_LABELS:
            continue
        cdf_name = payload.get('per_ue_se_cdf_file')
        if not cdf_name:
            continue
        cdf_path = result_path.parent / cdf_name
        with np.load(cdf_path) as archive:
            x = np.asarray(archive['rate_mbps'], dtype=float)
            y = np.asarray(archive['rate_cdf'], dtype=float)
        baselines[policy] = {'label': POLICY_LABELS[policy], 'x': x, 'y': y, 'policy_type': payload.get('policy_type'), 'cluster_policy': payload.get('cluster_policy'), 'sum_rate_mbps': float(payload['sum_rate_mbps']), 'qos_violation_rate': float(payload['qos_violation_rate']), 'requirement_mbps': float(payload['min_rate_requirement_mbps'])}
    return baselines

def load_learned(root: Path, tag: str, families: list[str], seeds: list[int], run_name: str) -> tuple[dict, list, float]:
    learned = {}
    all_curves = []
    requirements = []
    for family in families:
        family_curves = []
        for seed in seeds:
            parent = root / f'{tag}_{family}_s{seed}'
            run_dir = latest_run_dir(parent, None if run_name == 'latest' else run_name)
            (x, y, threshold, _floor) = load_cdf(run_dir, 'rate')
            family_curves.append((x, y))
            all_curves.append((x, y))
            requirements.append(threshold)
        learned[family] = family_curves
    if not requirements or not np.allclose(requirements, requirements[0]):
        raise ValueError('Learned runs use inconsistent rate requirements')
    return (learned, all_curves, float(requirements[0]))

def common_support(curves: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    upper = max((float(x[np.searchsorted(y, 0.999, side='left')]) for (x, y) in curves))
    return np.linspace(0.0, upper, 1200)

def mean_cdf(curves: list[tuple[np.ndarray, np.ndarray]], support: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    samples = np.stack([np.interp(support, x, y) for (x, y) in curves])
    mean = np.mean(samples, axis=0)
    if samples.shape[0] == 1:
        ci95 = np.zeros_like(mean)
    else:
        ci95 = 1.96 * np.std(samples, axis=0, ddof=1) / np.sqrt(samples.shape[0])
    return (mean, np.clip(ci95, 0.0, 1.0))

def plot_cdf(x: np.ndarray, y: np.ndarray, label: str, color: str, linestyle: str='-', marker: str | None=None, markevery: int | None=None) -> None:
    plt.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=2.0, marker=marker, markersize=4.5, markerfacecolor='white', markeredgewidth=1.2, markevery=markevery)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--seeds', default='2,3,4,5,6')
    parser.add_argument('--families', default='mlp,tf_actor')
    parser.add_argument('--result-root', default='results/IICMSMobility/PureMobility/mappo')
    parser.add_argument('--run', default='run1')
    parser.add_argument('--baseline-suite-dir', required=True)
    parser.add_argument('--out-prefix', required=True)
    parser.add_argument('--panel', choices=('both', 'left', 'right', 'single'), default='both', help='draw both panels, export one panel, or merge all learned, RSSI, and fixed-mode curves into one panel')
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(',') if value.strip()]
    families = [value.strip() for value in args.families.split(',') if value.strip()]
    unknown = set(families) - set(FAMILY_LABELS)
    if unknown:
        parser.error(f'Unknown learned families: {sorted(unknown)}')
    root = Path(args.result_root)
    (learned, learned_curves, requirement) = load_learned(root, args.tag, families, seeds, args.run)
    baseline_map = load_baselines(Path(args.baseline_suite_dir))
    required_baselines = {'dc_rssi', 'sc_rssi'}
    missing = required_baselines - set(baseline_map)
    if missing:
        raise FileNotFoundError(f'Missing RSSI baseline results: {sorted(missing)}')
    all_curves = list(learned_curves)
    all_curves.extend(((item['x'], item['y']) for item in baseline_map.values()))
    support = common_support(all_curves)
    best_fixed_by_group = {}
    for (policy, item) in baseline_map.items():
        mode = fixed_mode(policy)
        if mode is None:
            continue
        group = (str(item['cluster_policy']), mode)
        current = best_fixed_by_group.get(group)
        if current is None or item['sum_rate_mbps'] > current['sum_rate_mbps']:
            best_fixed_by_group[group] = {**item, 'policy': policy, 'mode': mode}
    show_left = args.panel in {'both', 'left', 'single'}
    show_right = args.panel in {'both', 'right', 'single'}
    if args.panel == 'both':
        (fig, axes) = plt.subplots(1, 2, figsize=(12.5, 5.4), dpi=180, gridspec_kw={'width_ratios': [1.15, 1.0]})
        (main_ax, fixed_ax) = axes
    elif args.panel == 'left':
        (fig, main_ax) = plt.subplots(figsize=(7.4, 5.4), dpi=180)
        fixed_ax = None
    elif args.panel == 'right':
        (fig, fixed_ax) = plt.subplots(figsize=(7.4, 5.4), dpi=180)
        main_ax = None
    else:
        (fig, single_ax) = plt.subplots(figsize=(9.6, 6.2), dpi=180)
        main_ax = single_ax
        fixed_ax = single_ax
    summary = {'tag': args.tag, 'seeds': seeds, 'requirement_mbps': requirement, 'learned': {}, 'baselines': {}}
    learned_colors = {'mlp': '#2563eb', 'tf_actor': '#ea580c', 'orig_mappo': '#15803d'}
    if show_left:
        for family in families:
            (mean, ci95) = mean_cdf(learned[family], support)
            color = learned_colors[family]
            main_ax.fill_between(support, np.clip(mean - ci95, 0.0, 1.0), np.clip(mean + ci95, 0.0, 1.0), color=color, alpha=0.13, linewidth=0)
            plot_cdf(support, mean, FAMILY_LABELS[family], color)
            summary['learned'][family] = {'seeds': len(seeds), 'cdf_at_requirement': float(np.interp(requirement, support, mean))}
    style = {'dc_rssi': ('#16a34a', '-'), 'sc_rssi': ('#7c3aed', '--')}
    if show_left:
        for policy in ('dc_rssi', 'sc_rssi'):
            item = baseline_map[policy]
            (color, linestyle) = style[policy]
            plot_cdf(support, np.interp(support, item['x'], item['y']), item['label'], color, linestyle)
            summary['baselines'][policy] = {'sum_rate_mbps': item['sum_rate_mbps'], 'cdf_at_requirement': float(np.interp(requirement, item['x'], item['y']))}
        main_ax.axvline(requirement, color='#444444', linestyle='--', linewidth=1.0, label=f'Requirement = {requirement:.1f} Mbps')
        if args.panel != 'single':
            main_ax.set_title('Learned policies vs RSSI baselines')
            main_ax.set_xlabel('Per-UE rate (Mbps)')
            main_ax.set_ylabel('CDF')
            main_ax.set_ylim(0.0, 1.0)
            main_ax.grid(True, alpha=0.22, linewidth=0.8)
            main_ax.legend(frameon=False, fontsize=8, loc='lower right')
    fixed_styles = {('static', 'NC'): ('#6D28D9', '-', 'o'), ('dynamic', 'NC'): ('#DB2777', '--', 's'), ('static', 'CB'): ('#15803D', '-', '^'), ('dynamic', 'CB'): ('#65A30D', '--', 'D'), ('static', 'JT'): ('#B91C1C', '-', 'v'), ('dynamic', 'JT'): ('#F59E0B', '--', 'P')}
    fixed_groups = [('static', mode) for mode in ('NC', 'CB', 'JT') if ('static', mode) in best_fixed_by_group] + [('dynamic', mode) for mode in ('NC', 'CB', 'JT') if ('dynamic', mode) in best_fixed_by_group]
    for (cluster_policy, mode) in fixed_groups:
        item = best_fixed_by_group[cluster_policy, mode]
        (color, linestyle, marker) = fixed_styles[cluster_policy, mode]
        plot_cdf(support, np.interp(support, item['x'], item['y']), item['label'], color, linestyle, marker=marker, markevery=80)
        summary['baselines'][item['policy']] = {'sum_rate_mbps': item['sum_rate_mbps'], 'cdf_at_requirement': float(np.interp(requirement, item['x'], item['y'])), 'cluster_policy': cluster_policy, 'mode': mode}
    if show_right:
        if args.panel != 'single':
            fixed_ax.axvline(requirement, color='#444444', linestyle='--', linewidth=1.0, label=f'Requirement = {requirement:.1f} Mbps')
            fixed_ax.set_title('Best cluster size per fixed mode and clustering')
            fixed_ax.set_xlabel('Per-UE rate (Mbps)')
            fixed_ax.set_ylabel('CDF')
            fixed_ax.set_ylim(0.0, 1.0)
            fixed_ax.grid(True, alpha=0.22, linewidth=0.8)
            fixed_ax.legend(frameon=False, fontsize=7.5, loc='lower right', ncol=2)
    if args.panel == 'single':
        single_ax.set_title('Per-UE rate CDF: learned policies and fixed baselines')
        single_ax.set_xlabel('Per-UE rate (Mbps)')
        single_ax.set_ylabel('CDF')
        single_ax.set_ylim(0.0, 1.0)
        single_ax.grid(True, alpha=0.22, linewidth=0.8)
        single_ax.legend(frameon=False, fontsize=7, loc='lower right', ncol=2)
    if args.panel == 'both':
        fig.suptitle(f'Per-UE rate CDF comparison - {args.tag}', fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_prefix.with_suffix('.png')
    json_path = out_prefix.with_suffix('.json')
    fig.savefig(png_path, bbox_inches='tight')
    plt.close(fig)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'FIGURE={png_path.resolve()}')
    print(f'SUMMARY={json_path.resolve()}')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
