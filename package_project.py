# -*- coding: utf-8 -*-
"""One-command source/results packaging with tests and SHA256 manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


EXCLUDED_PARTS = {
    "__pycache__",
    "results",
    "logs_iicms",
    "review_bundles",
    "release_bundles",
    "tmp",
    "analysis_outputs",
}
EXCLUDED_PREFIXES = (
    "results_eval_",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_sources(root: Path) -> list[Path]:
    paths = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if any(
            part.startswith(EXCLUDED_PREFIXES)
            for part in relative.parts
        ):
            continue
        if path.suffix in {".py", ".sh", ".ps1"} or path.name in {
            "README.md", "requirements.txt", "channel.npz"
        }:
            paths.append(path)
    return sorted(paths)


def write_manifest(root: Path, paths: list[Path]) -> Path:
    manifest_path = root / "SOURCE_SHA256.json"
    manifest = {
        path.relative_to(root).as_posix(): sha256(path).upper()
        for path in paths
        if path.name != manifest_path.name
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def included_project_files(root: Path) -> list[Path]:
    paths = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if any(
            part.startswith(EXCLUDED_PREFIXES)
            for part in relative.parts
        ):
            continue
        if path.suffix == ".pyc":
            continue
        if relative.parts[:2] == ("screening", "results"):
            continue
        paths.append(path)
    return sorted(paths)


def write_zip(
    root: Path,
    destination: Path,
    paths: list[Path],
) -> None:
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for path in paths:
            arcname = (
                Path(root.name) / path.relative_to(root)
            ).as_posix()
            archive.write(path, arcname)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="update")
    parser.add_argument("--output-dir", default="release_bundles")
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    if args.run_tests:
        subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            cwd=root,
            check=True,
        )

    sources = tracked_sources(root)
    write_manifest(root, sources)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_prefix = root.name

    code_path = output_dir / (
        f"{archive_prefix}_{args.tag}_{timestamp}.zip"
    )
    write_zip(root, code_path, included_project_files(root))

    results_root = root / "screening" / "results"
    result_files = [
        root / "README.md",
        root / "SOURCE_SHA256.json",
    ]
    if results_root.exists():
        result_files.extend(
            path for path in results_root.rglob("*") if path.is_file()
        )
    main_results_root = root / "results"
    if main_results_root.exists():
        result_files.extend(
            path for path in main_results_root.rglob("*")
            if path.is_file()
        )
    analysis_root = root / "analysis_outputs"
    if analysis_root.exists():
        result_files.extend(
            path for path in analysis_root.rglob("*") if path.is_file()
        )
    result_path = output_dir / (
        f"{archive_prefix}_{args.tag}_results_{timestamp}.zip"
    )
    write_zip(root, result_path, sorted(set(result_files)))

    latest_code = output_dir / f"{archive_prefix}_latest.zip"
    latest_results = output_dir / (
        f"{archive_prefix}_latest_results.zip"
    )
    shutil.copyfile(code_path, latest_code)
    shutil.copyfile(result_path, latest_results)

    print(f"CODE_ZIP={code_path}")
    print(f"CODE_SHA256={sha256(code_path).upper()}")
    print(f"RESULTS_ZIP={result_path}")
    print(f"RESULTS_SHA256={sha256(result_path).upper()}")
    print(f"LATEST_CODE={latest_code}")
    print(f"LATEST_RESULTS={latest_results}")


if __name__ == "__main__":
    main()
