from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
CORE_FILENAMES = {'config.json', 'source_sha256.json', 'checkpoint_selection.json', 'curve.csv', 'metrics_history.json', 'summary.json', 'validation_curve.csv', 'validation_history.json', 'validation_convergence.json'}
BEST_MODEL_FILENAMES = {'actor_best.pt', 'critic_best.pt', 'actor.pt', 'critic.pt'}

def require_tag(value: str) -> str:
    tag = str(value).strip()
    if not tag:
        raise ValueError('--tag must not be empty')
    return tag

def is_useful_result(path: Path, mode: str) -> bool:
    if mode == 'full':
        return True
    parts = path.parts
    name = path.name
    if name in CORE_FILENAMES:
        return True
    if path.suffix == '.log' and 'logs_iicms' in parts:
        return True
    if path.suffix == '.json' and any((part.startswith(('evaluation_', 'baseline_')) for part in parts)):
        return True
    if path.suffix == '.npz' and any((part.startswith(('evaluation_', 'baseline_suite')) for part in parts)):
        return True
    if name in {'baseline_summary.csv', 'baseline_summary.json', 'rssi_threshold_calibration.json', 'rssi_threshold_calibration.csv', 'selected_rssi_thresholds.env'}:
        return True
    if any((part.endswith('_figures') for part in parts)):
        if path.suffix in {'.json', '.png', '.svg', '.pdf'}:
            return True
    if mode == 'best':
        parent = path.parent.name
        if parent == 'models' and (name.startswith('actor_agent') or name.startswith('critic_agent')) and name.endswith('.pt'):
            return True
        if parent == 'models' and name in BEST_MODEL_FILENAMES:
            if name in {'actor_best.pt', 'critic_best.pt'}:
                return True
            actor_best = path.parent / 'actor_best.pt'
            critic_best = path.parent / 'critic_best.pt'
            return not actor_best.exists() and (not critic_best.exists())
    return False

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--log-dir', default='logs_iicms')
    parser.add_argument('--output-dir', default='review_bundles')
    parser.add_argument('--mode', choices=('results', 'best', 'full'), default='results', help='results keeps JSON/log summaries only; best also keeps best actor/critic weights; full keeps all snapshots')
    parser.add_argument('--include-tag', action='append', default=[], help='additional completed tag to include for direct comparison')
    args = parser.parse_args()
    try:
        tag = require_tag(args.tag)
    except ValueError as error:
        parser.error(str(error))
    root = Path(__file__).resolve().parent
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f'{tag}_review_{stamp}.zip'
    manifest = []
    package_tags = [item for item in dict.fromkeys([tag, *args.include_tag]) if item.strip()]
    with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as bundle:
        candidates = []
        for package_tag in package_tags:
            candidates.extend((root / args.log_dir).glob(f'{package_tag}*'))
        results_root = root / 'results' / 'IICMSMobility'
        for package_tag in package_tags:
            for run_path in results_root.rglob(f'{package_tag}*'):
                if run_path.is_dir():
                    candidates.extend(run_path.rglob('*'))
        included_bytes = 0
        excluded_files = 0
        excluded_bytes = 0
        for path in sorted(set(candidates)):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if not is_useful_result(relative, args.mode):
                excluded_files += 1
                excluded_bytes += path.stat().st_size
                continue
            bundle.write(path, relative.as_posix())
            included_bytes += path.stat().st_size
            manifest.append({'path': relative.as_posix(), 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        summary = {'mode': args.mode, 'tags': package_tags, 'included_files': len(manifest), 'included_bytes': included_bytes, 'excluded_files': excluded_files, 'excluded_bytes': excluded_bytes}
        bundle.writestr('BUNDLE_MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        bundle.writestr('BUNDLE_SUMMARY.json', json.dumps(summary, ensure_ascii=False, indent=2))
    print(str(archive))
if __name__ == '__main__':
    main()
