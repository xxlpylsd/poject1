#!/usr/bin/env python3
"""Create publication-oriented views from AP-count sweep CSV results."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


LABELS = {
    "mlp": "MLP-factorized (new)",
    "old_mlp": "MLP-flat (old)",
    "tf": "TF-native",
    "dc_rssi": "DC-RSSI",
    "sc_rssi": "SC-RSSI",
    "dc_cb_l5": "DC-CB-L5",
    "sc_cb_l5": "SC-CB-L5",
    "dc_jt_l2": "DC-JT-L2",
    "sc_jt_l3": "SC-JT-L3",
    "dc_nc": "DC-NC",
    "sc_nc": "SC-NC",
}
STYLES = {
    "mlp": ("#2563EB", "o"),
    "old_mlp": ("#7C3AED", "D"),
    "tf": ("#EA580C", "s"),
    "dc_rssi": ("#16A34A", "^"),
    "dc_cb_l5": ("#0891B2", "v"),
}


def read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def ci_radius(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    critical = 1.96
    try:
        from scipy.stats import t as student_t

        critical = float(
            student_t.ppf(0.975, df=len(values) - 1)
        )
    except Exception:
        critical = 2.571 if len(values) == 6 else 1.96
    return critical * statistics.stdev(values) / math.sqrt(
        len(values)
    )


def grouped_rows(rows: list[dict]) -> dict[tuple[str, int], list[dict]]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["strategy"]), int(row["ap_count"])), []
        ).append(row)
    return grouped


def series(
    grouped: dict[tuple[str, int], list[dict]],
    strategy: str,
    metric: str,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = []
    for (name, ap_count), group in grouped.items():
        if name != strategy:
            continue
        values = [
            float(row[metric]) * scale for row in group
        ]
        points.append((
            ap_count,
            statistics.mean(values),
            ci_radius(values),
        ))
    points.sort(key=lambda item: item[0])
    return tuple(np.asarray([point[index] for point in points])
                 for index in range(3))  # type: ignore[return-value]


def paired_differences(
    grouped: dict[tuple[str, int], list[dict]],
    ap_counts: list[int],
    metric: str,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    means = []
    radii = []
    win_counts = []
    for ap_count in ap_counts:
        tf_rows = {
            int(row["model_seed"]): float(row[metric]) * scale
            for row in grouped.get(("tf", ap_count), [])
            if row.get("model_seed") not in {None, ""}
        }
        mlp_rows = {
            int(row["model_seed"]): float(row[metric]) * scale
            for row in grouped.get(("mlp", ap_count), [])
            if row.get("model_seed") not in {None, ""}
        }
        seeds = sorted(set(tf_rows) & set(mlp_rows))
        differences = [
            tf_rows[seed] - mlp_rows[seed] for seed in seeds
        ]
        if not differences:
            raise ValueError(
                f"No paired TF/MLP seeds at AP count {ap_count}"
            )
        means.append(statistics.mean(differences))
        radii.append(ci_radius(differences))
        win_counts.append(sum(value > 0.0 for value in differences))
    return (
        np.asarray(ap_counts, dtype=float),
        np.asarray(means),
        np.asarray(radii),
        win_counts,
    )


def relative_paired_gain(
    grouped: dict[tuple[str, int], list[dict]],
    ap_counts: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    means = []
    radii = []
    for ap_count in ap_counts:
        tf_rows = {
            int(row["model_seed"]): float(row["sum_rate_mbps"])
            for row in grouped.get(("tf", ap_count), [])
            if row.get("model_seed") not in {None, ""}
        }
        mlp_rows = {
            int(row["model_seed"]): float(row["sum_rate_mbps"])
            for row in grouped.get(("mlp", ap_count), [])
            if row.get("model_seed") not in {None, ""}
        }
        seeds = sorted(set(tf_rows) & set(mlp_rows))
        values = [
            100.0
            * (tf_rows[seed] - mlp_rows[seed])
            / mlp_rows[seed]
            for seed in seeds
        ]
        means.append(statistics.mean(values))
        radii.append(ci_radius(values))
    return np.asarray(means), np.asarray(radii)


def setup_axis(axis, ap_counts: list[int], title: str, ylabel: str):
    axis.set_xticks(ap_counts)
    axis.set_xlabel("Number of APs")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.22, linewidth=0.8)


def plot_main_comparison(
    rows: list[dict],
    out_path: Path,
) -> None:
    grouped = grouped_rows(rows)
    strategies = [
        strategy for strategy in (
            "mlp", "old_mlp", "tf", "dc_rssi", "dc_cb_l5"
        )
        if any(name == strategy for name, _ap in grouped)
    ]
    ap_counts = sorted({
        int(row["ap_count"]) for row in rows
    })
    panels = (
        ("sum_rate_mbps", 1.0, "Total sum rate (Mbps)"),
        ("rate_per_ap_mbps", 1.0, "Sum rate per AP (Mbps/AP)"),
        ("per_ue_rate_p05_mbps", 1.0, "Per-UE rate P05 (Mbps)"),
        ("qos_violation_rate", 100.0, "QoS violation rate (%)"),
    )
    fig, axes = plt.subplots(
        2, 2, figsize=(12.2, 7.8), dpi=180
    )
    for axis, (metric, scale, ylabel) in zip(
        axes.ravel(), panels
    ):
        for strategy in strategies:
            x, mean, radius = series(
                grouped, strategy, metric, scale
            )
            color, marker = STYLES[strategy]
            if np.any(radius > 0.0):
                axis.fill_between(
                    x,
                    mean - radius,
                    mean + radius,
                    color=color,
                    alpha=0.11,
                    linewidth=0,
                )
            axis.plot(
                x,
                mean,
                color=color,
                marker=marker,
                linewidth=2.2,
                markersize=5.5,
                label=LABELS[strategy],
            )
        setup_axis(
            axis,
            ap_counts,
            ylabel.replace(" (%)", ""),
            ylabel,
        )
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(5, len(strategies)),
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Zero-shot AP-count scalability: representative strategies",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_advantage(
    rows: list[dict],
    out_path: Path,
) -> dict:
    grouped = grouped_rows(rows)
    ap_counts = sorted({
        int(row["ap_count"]) for row in rows
    })
    x, rate_delta, rate_ci, wins = paired_differences(
        grouped, ap_counts, "sum_rate_mbps"
    )
    gain_pct, gain_pct_ci = relative_paired_gain(
        grouped, ap_counts
    )
    _x, p05_delta, p05_ci, _ = paired_differences(
        grouped, ap_counts, "per_ue_rate_p05_mbps"
    )
    _x, qos_delta, qos_ci, _ = paired_differences(
        grouped, ap_counts, "qos_violation_rate", 100.0
    )
    panels = (
        (
            rate_delta,
            rate_ci,
            "TF - MLP system rate (Mbps)",
            "Absolute total-rate gain",
        ),
        (
            gain_pct,
            gain_pct_ci,
            "TF vs MLP rate gain (%)",
            "Relative total-rate gain",
        ),
        (
            p05_delta,
            p05_ci,
            "TF - MLP P05 (Mbps)",
            "Low-rate-tail improvement",
        ),
        (
            qos_delta,
            qos_ci,
            "TF - MLP QoS violation (percentage points)",
            "QoS improvement (negative is better)",
        ),
    )
    fig, axes = plt.subplots(
        2, 2, figsize=(12.2, 7.8), dpi=180
    )
    for index, (axis, (values, radii, ylabel, title)) in enumerate(
        zip(axes.ravel(), panels)
    ):
        colors = [
            "#16A34A" if value > 0 else "#DC2626"
            for value in values
        ]
        if index == 3:
            colors = [
                "#16A34A" if value < 0 else "#DC2626"
                for value in values
            ]
        axis.bar(
            x,
            values,
            width=4.2,
            color=colors,
            alpha=0.72,
            yerr=radii,
            capsize=4,
            edgecolor="black",
            linewidth=0.7,
        )
        axis.axhline(
            0.0, color="#333333", linewidth=1.0
        )
        setup_axis(axis, ap_counts, title, ylabel)
        if index == 0:
            for ap, value, win_count in zip(
                x, values, wins
            ):
                axis.annotate(
                    f"{win_count}/6",
                    (ap, value),
                    textcoords="offset points",
                    xytext=(0, 8 if value >= 0 else -16),
                    ha="center",
                    fontsize=8,
                )
    fig.suptitle(
        "TF-native advantage over MLP-factorized by AP count",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "ap_counts": ap_counts,
        "tf_minus_mlp_rate_mbps": rate_delta.tolist(),
        "tf_minus_mlp_rate_ci95": rate_ci.tolist(),
        "tf_vs_mlp_rate_gain_pct": gain_pct.tolist(),
        "tf_vs_mlp_rate_gain_pct_ci95": gain_pct_ci.tolist(),
        "tf_minus_mlp_p05_mbps": p05_delta.tolist(),
        "tf_minus_mlp_p05_ci95": p05_ci.tolist(),
        "tf_minus_mlp_qos_pp": qos_delta.tolist(),
        "tf_minus_mlp_qos_ci95_pp": qos_ci.tolist(),
        "tf_wins_out_of_6": wins,
    }


def render_improved_plots(results_dir: Path) -> tuple[Path, Path]:
    rows = read_rows(results_dir / "raw_results.csv")
    main_path = results_dir / "ap_count_main_comparison.png"
    gain_path = results_dir / "ap_count_tf_vs_mlp_advantage.png"
    plot_main_comparison(rows, main_path)
    summary = plot_advantage(rows, gain_path)
    (results_dir / "ap_count_advantage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return main_path, gain_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        required=True,
        help="directory containing raw_results.csv",
    )
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    main_path, gain_path = render_improved_plots(results_dir)
    print(f"MAIN_FIGURE={main_path.resolve()}")
    print(f"ADVANTAGE_FIGURE={gain_path.resolve()}")


if __name__ == "__main__":
    main()
