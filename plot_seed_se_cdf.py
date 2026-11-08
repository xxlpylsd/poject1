# -*- coding: utf-8 -*-
"""Plot a multi-seed mean per-UE spectral-efficiency CDF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


FAMILY_LABELS = {
    "mlp": "MAPPO-MLP",
    "tf_actor": "MAPPO-Transformer",
    "orig_mappo": "Original-paper MAPPO",
}
FAMILY_STYLES = {
    "mlp": {"color": "#276FBF", "marker": "o"},
    "tf_actor": {"color": "#D97706", "marker": "s"},
    "orig_mappo": {"color": "#15803D", "marker": "^"},
}


def latest_run_dir(parent: Path, requested: str | None) -> Path:
    if requested:
        run_dir = parent / requested
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        return run_dir
    candidates = sorted(
        (path for path in parent.glob("run*") if path.is_dir()),
        key=lambda path: int(path.name[3:]),
    )
    if not candidates:
        raise FileNotFoundError(f"No run directories under {parent}")
    return candidates[-1]


def evaluation_file(run_dir: Path) -> Path:
    candidates = sorted(
        (run_dir / "evaluation_deterministic").glob("*.json")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No deterministic evaluation JSON under {run_dir}"
        )
    return candidates[-1]


def load_cdf(
    run_dir: Path,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    result_path = evaluation_file(run_dir)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    cdf_data = payload.get("per_ue_se_cdf")
    if (
        isinstance(cdf_data, dict)
        and "se_bps_hz" in cdf_data
        and "cdf" in cdf_data
    ):
        # Backward compatibility for the first CDF export format.
        x = np.asarray(cdf_data["se_bps_hz"], dtype=float)
        y = np.asarray(cdf_data["cdf"], dtype=float)
        if metric == "rate":
            rate_scale = (
                float(payload["min_rate_requirement_mbps"])
                / float(payload["rate_requirement_se_bps_hz"])
            )
            x = x * rate_scale
    else:
        cdf_name = payload.get("per_ue_se_cdf_file")
        if not cdf_name:
            raise ValueError(
                f"{run_dir} has no per-UE CDF. Re-run "
                "evaluate_iicms.py with the updated evaluator."
            )
        with np.load(result_path.parent / cdf_name) as archive:
            if metric == "rate":
                x = np.asarray(archive["rate_mbps"], dtype=float)
                y = np.asarray(archive["rate_cdf"], dtype=float)
            else:
                x = np.asarray(archive["se_bps_hz"], dtype=float)
                y = np.asarray(archive["cdf"], dtype=float)
    if metric == "rate":
        threshold = float(payload["min_rate_requirement_mbps"])
        floor = float(payload.get("min_ue_rate_mbps", 0.0))
    else:
        threshold = float(payload["rate_requirement_se_bps_hz"])
        floor = float(payload.get("min_ue_se_bps_hz", 0.0))
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError(f"Malformed CDF in {run_dir}")
    if np.any(np.diff(x) < 0.0) or np.any(np.diff(y) < -1e-12):
        raise ValueError(f"Non-monotone CDF in {run_dir}")
    return x, y, threshold, floor


def interpolate_cdfs(
    curves: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Average CDFs seed-by-seed on one common support."""
    stable_upper = max(
        float(x[np.searchsorted(y, 0.999, side="left")])
        for x, y in curves
    )
    support = np.linspace(0.0, stable_upper, 800)
    interpolated = np.stack([
        np.interp(support, x, y)
        for x, y in curves
    ])
    return support, interpolated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a multi-seed mean SE CDF from evaluation JSON."
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--result-root",
        default="results/IICMSMobility/PureMobility/mappo",
    )
    parser.add_argument("--seeds", default="2,3,4,5,6,7,8")
    parser.add_argument("--families", default="mlp,tf_actor")
    parser.add_argument(
        "--metric",
        choices=("se", "rate"),
        default="se",
    )
    parser.add_argument(
        "--run",
        default="run1",
        help="specific run directory; use 'latest' for automatic selection",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.result_root)
    seeds = [
        int(value.strip())
        for value in args.seeds.split(",")
        if value.strip()
    ]
    families = [
        value.strip()
        for value in args.families.split(",")
        if value.strip()
    ]
    unknown = set(families) - set(FAMILY_LABELS)
    if unknown:
        parser.error(f"Unknown families: {sorted(unknown)}")

    requested = None if args.run == "latest" else args.run
    curves: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    thresholds = []
    floors = []
    for family in families:
        curves[family] = []
        for seed in seeds:
            parent = root / f"{args.tag}_{family}_s{seed}"
            run_dir = latest_run_dir(parent, requested)
            x, y, threshold, floor = load_cdf(
                run_dir, args.metric
            )
            curves[family].append((x, y))
            thresholds.append(threshold)
            floors.append(floor)
    if not np.allclose(thresholds, thresholds[0]):
        raise ValueError(
            "Evaluation runs use inconsistent SE rate requirements"
        )
    if not np.allclose(floors, floors[0]):
        raise ValueError(
            "Evaluation runs use inconsistent UE SE floors"
        )

    all_curves = [
        curve for family in families for curve in curves[family]
    ]
    support, _ = interpolate_cdfs(all_curves)
    summary = {
        "tag": args.tag,
        "seeds": seeds,
        "metric": args.metric,
        "requirement": thresholds[0],
        "floor": floors[0],
        "families": {},
    }

    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=180)
    for family in families:
        samples = np.stack([
            np.interp(support, x, y)
            for x, y in curves[family]
        ])
        mean = np.mean(samples, axis=0)
        if samples.shape[0] > 1:
            ci95 = (
                1.96
                * np.std(samples, axis=0, ddof=1)
                / np.sqrt(samples.shape[0])
            )
        else:
            ci95 = np.zeros_like(mean)
        style = FAMILY_STYLES[family]
        ax.fill_between(
            support,
            np.clip(mean - ci95, 0.0, 1.0),
            np.clip(mean + ci95, 0.0, 1.0),
            color=style["color"],
            alpha=0.14,
            linewidth=0,
        )
        ax.plot(
            support,
            mean,
            color=style["color"],
            linewidth=2.2,
            label=FAMILY_LABELS[family],
        )
        cdf_at_requirement = float(np.interp(
            thresholds[0], support, mean
        ))
        summary["families"][family] = {
            "seeds": len(seeds),
            "mean_cdf_at_requirement": cdf_at_requirement,
        }

    ax.axvline(
        thresholds[0],
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label=(
            f"$R_{{req}}$ = {thresholds[0]:.1f} Mbps"
            if args.metric == "rate"
            else f"$R_{{req}}$ = {thresholds[0]:.1f} bps/Hz"
        ),
    )
    if floors[0] > 0.0:
        ax.axvline(
            floors[0],
            color="#8A8A8A",
            linestyle=":",
            linewidth=1.1,
            label=(
                f"Rate floor = {floors[0]:.1f} Mbps"
                if args.metric == "rate"
                else f"SE floor = {floors[0]:.2f} bps/Hz"
            ),
        )
    ax.set_xlim(0.0, float(support[-1]))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(
        "Per-UE rate (Mbps)"
        if args.metric == "rate"
        else "Per-UE spectral efficiency (bps/Hz)"
    )
    ax.set_ylabel("CDF")
    ax.set_title(
        f"{len(seeds)}-seed mean per-UE "
        f"{'rate' if args.metric == 'rate' else 'SE'} CDF"
    )
    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    summary_path = out.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"FIGURE={out.resolve()}")
    print(f"SUMMARY={summary_path.resolve()}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
