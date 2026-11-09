from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--out-prefix', default=None)
    args = parser.parse_args()
    root = Path(args.root)
    paths = sorted(root.glob('*/*.json'))
    if not paths:
        raise SystemExit(f'No baseline result JSONs found under {root}')
    rows = []
    for path in paths:
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('policy_type') not in {'fixed_action', 'rssi_threshold'}:
            continue
        mode_share = data.get('mode_share', {})
        size_share = data.get('cluster_size_share_cb_jt', {})
        rows.append({'policy': data.get('policy_label'), 'policy_type': data.get('policy_type'), 'cluster_policy': data.get('cluster_policy'), 'fixed_action': data.get('fixed_action'), 'rssi_cluster_size': data.get('rssi_cluster_size'), 'rssi_threshold_hi': data.get('rssi_threshold_hi'), 'rssi_threshold_lo': data.get('rssi_threshold_lo'), 'sum_rate_mbps': data.get('sum_rate_mbps'), 'system_se_bps_hz': data.get('time_averaged_system_se_bps_hz'), 'p05_mbps': data.get('per_ue_rate_p05_mbps'), 'qos_violation_rate': data.get('qos_violation_rate'), 'overload_ratio': data.get('overload_ratio'), 'fronthaul_drop_ratio': data.get('fronthaul_drop_ratio'), 'fronthaul_queue_delay_ms': data.get('fronthaul_queue_delay_ms'), 'cluster_change_rate': data.get('cluster_change_rate'), 'mode_JT': mode_share.get('JT'), 'mode_CB': mode_share.get('CB'), 'mode_NC': mode_share.get('NC'), 'cluster_size_2': size_share.get('2'), 'cluster_size_3': size_share.get('3'), 'cluster_size_4': size_share.get('4'), 'cluster_size_5': size_share.get('5'), 'result_json': str(path)})
    if not rows:
        raise SystemExit(f'No baseline-policy results found under {root}')
    prefix = Path(args.out_prefix) if args.out_prefix else root / 'summary'
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix('.json')
    csv_path = prefix.with_suffix('.csv')
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    with csv_path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'SUMMARY_JSON={json_path}')
    print(f'SUMMARY_CSV={csv_path}')
if __name__ == '__main__':
    main()
