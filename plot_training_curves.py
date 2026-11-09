# -*- coding: utf-8 -*-
"""Plot multi-seed smoothed training curves and seed variability."""
from __future__ import annotations

import argparse
import csv
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
    "mlp": {"color": "#276FBF"},
    "tf_actor": {"color": "#D97706"},
    "orig_mappo": {"color": "#15803D"},
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


def load_curve(
    run_dir: Path,
    metric: str,
    source: str = "training",
) -> tuple[np.ndarray, np.ndarray]:
    filename = (
        "validation_curve.csv"
        if source == "validation"
        else "curve.csv"
    )
    path = run_dir / "logs" / filename
    if not path.exists():
        raise FileNotFoundError(path)
    steps = []
    values = []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            steps.append(int(row["environment_steps"]))
            if metric == "rate":
                column = "sum_rate_mbps"
            elif metric == "p05":
                column = "p05_rate_mbps"
            elif metric == "qos":
                column = "qos_violation_rate"
            elif source == "validation":
                column = "system_se_bps_hz"
            else:
                column = "system_sum_se_bps_hz"
            values.append(float(row[column]))
    if not steps:
        raise ValueError(f"Empty training curve: {path}")
    return np.asarray(steps), np.asarray(values)


def centered_smooth(values: np.ndarray, radius: int = 1) -> np.ndarray:
    output = np.empty_like(values, dtype=float)
    for index in range(values.size):
        low = max(0, index - radius)
        high = min(values.size, index + radius + 1)
        output[index] = float(np.mean(values[low:high]))
    return output


def load_convergence(run_dir: Path) -> dict | None:
    path = run_dir / "logs" / "validation_convergence.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def interpolate_curves(
    curves: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    lower = max(float(steps[0]) for steps, _values in curves)
    upper = min(float(steps[-1]) for steps, _values in curves)
    if upper < lower:
        raise ValueError("Training curves have no overlapping range")
    support = (
        np.asarray([lower], dtype=float)
        if np.isclose(upper, lower)
        else np.linspace(lower, upper, 600)
    )
    rows = np.stack([
        np.interp(support, steps, values)
        for steps, values in curves
    ])
    return support, rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot multi-seed smoothed training curves."
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--result-root",
        default="results/IICMSMobility/PureMobility/mappo",
    )
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--families", default="mlp,tf_actor")
    parser.add_argument(
        "--metric",
        choices=("se", "rate", "p05", "qos"),
        default="se",
    )
    parser.add_argument(
        "--source",
        choices=("training", "validation"),
        default="training",
        help=(
            "plot stochastic training rollouts or fixed-seed deterministic "
            "validation"
        ),
    )
    parser.add_argument("--run", default="latest")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

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

    root = Path(args.result_root)
    requested = None if args.run == "latest" else args.run
    curves = {}
    for family in families:
        curves[family] = []
        for seed in seeds:
            parent = root / f"{args.tag}_{family}_s{seed}"
            curves[family].append(
                load_curve(
                    latest_run_dir(parent, requested),
                    args.metric,
                    args.source,
                )
            )

    support, sample_rows = interpolate_curves([
        curve for family in families for curve in curves[family]
    ])
    summary = {
        "tag": args.tag,
        "metric": args.metric,
        "source": args.source,
        "seeds": seeds,
        "smoothing_radius_points": 1,
        "families": {},
    }
    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=180)

    for family in families:
        _family_support, rows = interpolate_curves(curves[family])
        mean = np.mean(rows, axis=0)
        smooth = centered_smooth(mean)
        if rows.shape[0] > 1:
            band = (
                1.96
                * np.std(rows, axis=0, ddof=1)
                / np.sqrt(rows.shape[0])
            )
        else:
            band = np.zeros_like(mean)
        style = FAMILY_STYLES[family]
        ax.fill_between(
            support,
            mean - band,
            mean + band,
            color=style["color"],
            alpha=0.13,
            linewidth=0,
        )
        ax.plot(
            support,
            smooth,
            color=style["color"],
            linewidth=2.2,
            label=FAMILY_LABELS[family],
        )
        summary["families"][family] = {
            "seeds": int(rows.shape[0]),
            "first_smoothed_value": float(smooth[0]),
            "last_smoothed_value": float(smooth[-1]),
            "last_five_mean_value": float(np.mean(mean[-5:])),
            "convergence_by_seed": [
                {
                    "seed": seed,
                    "report": load_convergence(
                        latest_run_dir(
                            root / f"{args.tag}_{family}_s{seed}",
                            requested,
                        )
                    ),
                }
                for seed in seeds
            ] if args.source == "validation" else [],
        }

    ax.set_xlabel("Environment steps")
    ax.set_ylabel(
        {
            "rate": "System rate (Mbps)",
            "se": "System SE (bps/Hz)",
            "p05": "P05 UE rate (Mbps)",
            "qos": "QoS violation rate",
        }[args.metric]
    )
    metric_label = {
        "rate": "rate",
        "se": "SE",
        "p05": "P05 rate",
        "qos": "QoS violation",
    }[args.metric]
    ax.set_title(
        f"{len(seeds)}-seed mean "
        f"{'validation' if args.source == 'validation' else 'training'} "
        f"{metric_label}"
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


if __name__ == "__main__":
    main()
