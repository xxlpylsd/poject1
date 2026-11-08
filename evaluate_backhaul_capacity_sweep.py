#!/usr/bin/env python3
"""Zero-shot average-backhaul-capacity sweep for trained policies."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from evaluate_ap_count_sweep import (  # noqa: E402
    BASELINE_LABELS,
    BASELINE_SPECS,
    DEFAULT_BASELINES,
    candidate_threshold_file,
    confidence_radius,
    execute_evaluation,
    find_result_json,
    parse_int_list,
    parse_str_list,
    read_env_file,
    run_dir_for_family,
    validate_reusable_result,
)
from plot_ap_count_results import LABELS, STYLES  # noqa: E402


DEFAULT_CAPACITIES_GBPS = "0.30,0.45,0.60,0.75,1.00,1.50"


def parse_float_list(value: str, name: str) -> list[float]:
    values = sorted({
        float(item.strip())
        for item in str(value).split(",")
        if item.strip()
    })
    if not values or any(item <= 0.0 for item in values):
        raise ValueError(f"{name} must contain positive values")
    return values


def capacity_label(center_gbps: float) -> str:
    return f"cap_{int(round(center_gbps * 1000)):04d}mbps"


def add_eval_args(
    command: list[str],
    run_dir: Path,
    output_dir: Path,
    center_gbps: float,
    variation_gbps: float,
    seed_start: int,
    episodes: int,
    horizon: int,
) -> tuple[int, int]:
    center = int(round(center_gbps * 1e9))
    half_width = int(round(variation_gbps * 1e9))
    low = max(0, center - half_width)
    high = center + half_width
    command.extend([
        "--run_dir", str(run_dir),
        "--seed_start", str(seed_start),
        "--episodes", str(episodes),
        "--horizon", str(horizon),
        "--phase", "test",
        "--backhaul_lo_override", str(low),
        "--backhaul_hi_override", str(high),
        "--out", str(output_dir),
    ])
    return low, high


def reusable_capacity_result(
    run_dir: Path,
    center_gbps: float,
    variation_gbps: float,
    seed_start: int,
    episodes: int,
    horizon: int,
) -> Path | None:
    eval_dir = run_dir / "evaluation_deterministic"
    if not eval_dir.is_dir():
        return None
    center = center_gbps * 1e9
    half_width = variation_gbps * 1e9
    expected_low = center - half_width
    expected_high = center + half_width
    for path in sorted(eval_dir.glob("*.json"), reverse=True):
        if not validate_reusable_result(
            path, seed_start, episodes, horizon
        ):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload.get("config", {})
        if (
            abs(float(config.get("backhaul_lo", -1.0))
                - expected_low) <= 1.0
            and abs(float(config.get("backhaul_hi", -1.0))
                    - expected_high) <= 1.0
        ):
            return path
    return None


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, float], list[dict]] = {}
    for row in rows:
        grouped.setdefault(
            (row["strategy"], float(row["capacity_gbps"])), []
        ).append(row)
    metrics = (
        "sum_rate_mbps",
        "per_ue_rate_p05_mbps",
        "qos_violation_rate",
        "overload_ratio",
        "fronthaul_queue_delay_ms",
        "cluster_change_rate",
    )
    result = []
    for (strategy, capacity), group in sorted(grouped.items()):
        row = {
            "strategy": strategy,
            "strategy_label": group[0]["strategy_label"],
            "capacity_gbps": capacity,
            "model_seed_count": len({
                item["model_seed"] for item in group
                if item["model_seed"] is not None
            }),
        }
        for metric in metrics:
            values = [float(item[metric]) for item in group]
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_ci95"] = confidence_radius(values)
        result.append(row)
    return result


def series(
    aggregate: list[dict],
    strategy: str,
    metric: str,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = sorted(
        (
            row for row in aggregate
            if row["strategy"] == strategy
        ),
        key=lambda row: row["capacity_gbps"],
    )
    return (
        np.asarray([
            row["capacity_gbps"] for row in selected
        ], dtype=float),
        np.asarray([
            row[f"{metric}_mean"] * scale for row in selected
        ], dtype=float),
        np.asarray([
            (row[f"{metric}_ci95"] or 0.0) * scale
            for row in selected
        ], dtype=float),
    )


def plot_all_capacities(
    aggregate: list[dict],
    capacities: list[float],
    output_path: Path,
) -> None:
    strategies = sorted({
        row["strategy"] for row in aggregate
    })
    fig, axis = plt.subplots(figsize=(9.4, 6.0), dpi=180)
    for strategy in strategies:
        x, mean, ci = series(
            aggregate, strategy, "sum_rate_mbps"
        )
        color, marker = STYLES.get(
            strategy, ("#555555", "o")
        )
        if np.any(ci > 0.0):
            axis.fill_between(
                x, mean - ci, mean + ci,
                color=color, alpha=0.09, linewidth=0,
            )
        axis.plot(
            x,
            mean,
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=5,
            label=LABELS.get(strategy, strategy),
        )
    axis.set_xticks(capacities)
    axis.set_xlabel("Average backhaul capacity per AP (Gbps)")
    axis.set_ylabel("Total sum rate (Mbps)")
    axis.set_title(
        "Zero-shot backhaul-capacity scalability"
    )
    axis.grid(True, alpha=0.22, linewidth=0.8)
    axis.legend(
        frameon=False, fontsize=7.5, ncol=2,
        loc="lower right",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_main_comparison(
    aggregate: list[dict],
    capacities: list[float],
    output_path: Path,
) -> None:
    strategies = [
        name for name in (
            "mlp", "old_mlp", "tf", "dc_rssi", "dc_cb_l5"
        )
        if any(row["strategy"] == name for row in aggregate)
    ]
    panels = (
        ("sum_rate_mbps", 1.0, "Total sum rate (Mbps)"),
        ("per_ue_rate_p05_mbps", 1.0, "Per-UE rate P05 (Mbps)"),
        ("qos_violation_rate", 100.0, "QoS violation rate (%)"),
        ("overload_ratio", 100.0, "Backhaul overload ratio (%)"),
    )
    fig, axes = plt.subplots(
        2, 2, figsize=(12.2, 7.8), dpi=180
    )
    for axis, (metric, scale, ylabel) in zip(
        axes.ravel(), panels
    ):
        for strategy in strategies:
            x, mean, ci = series(
                aggregate, strategy, metric, scale
            )
            color, marker = STYLES[strategy]
            if np.any(ci > 0.0):
                axis.fill_between(
                    x, mean - ci, mean + ci,
                    color=color, alpha=0.10, linewidth=0,
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
        axis.set_xticks(capacities)
        axis.set_xlabel("Average backhaul capacity per AP (Gbps)")
        axis.set_ylabel(ylabel)
        axis.set_title(ylabel.replace(" (%)", ""))
        axis.grid(True, alpha=0.22, linewidth=0.8)
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
        "Backhaul-capacity comparison: representative strategies",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_paper_figure5_style(
    aggregate: list[dict],
    capacities: list[float],
    output_path: Path,
) -> None:
    paper_labels = {
        "tf": "Proposed TF-native",
        "dc_rssi": "RSSI + DC",
        "sc_rssi": "RSSI + SC",
        "dc_jt_l2": "Only JT (DC-JT-L2)",
        "dc_cb_l5": "Only CB (DC-CB-L5)",
        "dc_nc": "Only NC (DC-NC)",
    }
    styles = {
        "tf": ("#EA580C", "s", "-"),
        "dc_rssi": ("#16A34A", "^", "-"),
        "sc_rssi": ("#7C3AED", "v", "--"),
        "dc_jt_l2": ("#B91C1C", "D", "-"),
        "dc_cb_l5": ("#0891B2", "o", "-"),
        "dc_nc": ("#6B7280", "P", "--"),
    }
    available = {
        row["strategy"] for row in aggregate
    }
    strategies = [
        strategy
        for strategy in paper_labels
        if strategy in available
    ]
    fig, axis = plt.subplots(
        figsize=(8.4, 5.6), dpi=180
    )
    for strategy in strategies:
        x, mean, ci = series(
            aggregate, strategy, "sum_rate_mbps"
        )
        color, marker, linestyle = styles[strategy]
        if np.any(ci > 0.0):
            axis.fill_between(
                x,
                mean - ci,
                mean + ci,
                color=color,
                alpha=0.10,
                linewidth=0,
            )
        axis.plot(
            x,
            mean,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.2,
            markersize=5.5,
            label=paper_labels[strategy],
        )
    axis.set_xticks(capacities)
    axis.set_xlabel(
        "Average backhaul capacity per AP (Gbps)"
    )
    axis.set_ylabel("Total sum rate (Mbps)")
    axis.set_title(
        "Effect of average AP backhaul capacity"
    )
    axis.grid(True, alpha=0.22, linewidth=0.8)
    axis.legend(
        frameon=False,
        fontsize=8.2,
        ncol=2,
        loc="lower right",
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate fixed checkpoints under different average AP "
            "backhaul capacities without retraining."
        )
    )
    parser.add_argument(
        "--result-root",
        default="results/IICMSMobility/PureMobility/mappo",
    )
    parser.add_argument(
        "--mlp-tag",
        default="fronthaul_queue_mlp_factorized_rate_400k",
    )
    parser.add_argument(
        "--old-mlp-tag",
        default="fronthaul_queue_native_rate_400k",
    )
    parser.add_argument(
        "--tf-tag",
        default="fronthaul_queue_native_rate_400k",
    )
    parser.add_argument("--mlp-seeds", default="1,2,3,4,5,6")
    parser.add_argument(
        "--old-mlp-seeds", default="1,2,3,4,5,6"
    )
    parser.add_argument("--tf-seeds", default="1,2,3,4,5,6")
    parser.add_argument(
        "--families", default="mlp,old_mlp,tf"
    )
    parser.add_argument(
        "--capacities-gbps",
        default=DEFAULT_CAPACITIES_GBPS,
        help="average per-AP backhaul capacity values in Gbps",
    )
    parser.add_argument(
        "--variation-gbps",
        type=float,
        default=0.05,
        help=(
            "half-width of the per-AP capacity variation; the reference "
            "0.6 Gbps point then matches 550-650 Mbps"
        ),
    )
    parser.add_argument(
        "--reference-capacity-gbps",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--baselines",
        default=DEFAULT_BASELINES,
    )
    parser.add_argument("--rssi-thresholds-env", default="")
    parser.add_argument("--rssi-cluster-size", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=93000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs/backhaul_capacity_zero_shot",
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--no-reuse-reference",
        dest="reuse_reference",
        action="store_false",
        default=True,
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.variation_gbps <= 0.0:
        parser.error("--variation-gbps must be positive")
    capacities = parse_float_list(
        args.capacities_gbps, "capacities"
    )
    families = parse_str_list(args.families)
    unknown_families = set(families) - {
        "mlp", "old_mlp", "tf"
    }
    if unknown_families:
        parser.error(
            f"Unknown families: {sorted(unknown_families)}"
        )
    baselines = [] if args.baselines.strip().lower() == "none" else (
        parse_str_list(args.baselines)
    )
    if set(baselines) - set(BASELINE_SPECS):
        parser.error(
            f"Unknown baselines: {sorted(set(baselines) - set(BASELINE_SPECS))}"
        )

    project_root = Path(__file__).resolve().parent
    result_root = (project_root / args.result_root).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_lists = {
        "mlp": parse_int_list(args.mlp_seeds, "MLP seeds"),
        "old_mlp": parse_int_list(
            args.old_mlp_seeds, "old MLP seeds"
        ),
        "tf": parse_int_list(args.tf_seeds, "TF seeds"),
    }
    tags = {
        "mlp": args.mlp_tag,
        "old_mlp": args.old_mlp_tag,
        "tf": args.tf_tag,
    }
    family_runs = {
        family: {
            seed: run_dir_for_family(
                result_root, tags[family], family, seed
            )
            for seed in seed_lists[family]
        }
        for family in families
    }
    for family, runs in family_runs.items():
        for seed, run_dir in runs.items():
            if not (run_dir / "config.json").exists():
                raise FileNotFoundError(
                    f"Missing {family} seed {seed} run: {run_dir}"
                )

    template_run = next(
        run for family in families
        for run in family_runs[family].values()
    )
    threshold_path = None
    if any(name.endswith("rssi") for name in baselines):
        if args.rssi_thresholds_env:
            threshold_path = candidate_threshold_file(
                args.rssi_thresholds_env, template_run
            )
        else:
            for family in families:
                for run_dir in family_runs[family].values():
                    candidate = candidate_threshold_file("", run_dir)
                    if candidate is not None:
                        threshold_path = candidate
                        break
                if threshold_path is not None:
                    break
        if threshold_path is None:
            parser.error(
                "RSSI baselines require a selected threshold env file"
            )
    thresholds = (
        read_env_file(threshold_path)
        if threshold_path is not None else {}
    )

    raw_rows: list[dict] = []
    result_paths: list[Path] = []
    for capacity_gbps in capacities:
        print(
            f"\n===== average backhaul {capacity_gbps:.2f} Gbps/AP =====",
            flush=True,
        )
        capacity_dir = output_dir / capacity_label(capacity_gbps)
        for family in families:
            for seed, run_dir in family_runs[family].items():
                eval_dir = capacity_dir / f"{family}_s{seed}"
                existing = None
                if (
                    args.reuse_reference
                    and abs(
                        capacity_gbps
                        - args.reference_capacity_gbps
                    ) < 1e-9
                ):
                    existing = reusable_capacity_result(
                        run_dir,
                        capacity_gbps,
                        args.variation_gbps,
                        args.seed_start,
                        args.episodes,
                        args.horizon,
                    )
                if existing is None:
                    command = [args.python_bin, "evaluate_iicms.py"]
                    low, high = add_eval_args(
                        command,
                        run_dir,
                        eval_dir,
                        capacity_gbps,
                        args.variation_gbps,
                        args.seed_start,
                        args.episodes,
                        args.horizon,
                    )
                    command.extend(["--checkpoint", "actor"])
                    existing_candidates = (
                        sorted(eval_dir.glob("*.json"))
                        if eval_dir.exists() else []
                    )
                    if args.skip_existing and existing_candidates:
                        result_path = existing_candidates[-1]
                    else:
                        execute_evaluation(
                            command,
                            project_root,
                            eval_dir / "evaluate.log",
                            args.dry_run,
                        )
                        if args.dry_run:
                            continue
                        result_path = find_result_json(eval_dir)
                else:
                    result_path = existing
                    low = int(round(
                        (capacity_gbps - args.variation_gbps) * 1e9
                    ))
                    high = int(round(
                        (capacity_gbps + args.variation_gbps) * 1e9
                    ))
                payload = json.loads(
                    result_path.read_text(encoding="utf-8")
                )
                raw_rows.append({
                    "strategy": family,
                    "strategy_label": LABELS[family],
                    "capacity_gbps": capacity_gbps,
                    "backhaul_lo_mbps": low / 1e6,
                    "backhaul_hi_mbps": high / 1e6,
                    "model_seed": seed,
                    "sum_rate_mbps": payload["sum_rate_mbps"],
                    "per_ue_rate_p05_mbps": payload[
                        "per_ue_rate_p05_mbps"
                    ],
                    "qos_violation_rate": payload[
                        "qos_violation_rate"
                    ],
                    "overload_ratio": payload["overload_ratio"],
                    "fronthaul_queue_delay_ms": payload[
                        "fronthaul_queue_delay_ms"
                    ],
                    "cluster_change_rate": payload[
                        "cluster_change_rate"
                    ],
                    "evaluation_json": str(result_path),
                })
                result_paths.append(result_path)

        for baseline in baselines:
            spec = BASELINE_SPECS[baseline]
            eval_dir = capacity_dir / "baselines" / baseline
            command = [args.python_bin, "evaluate_iicms.py"]
            low, high = add_eval_args(
                command,
                template_run,
                eval_dir,
                capacity_gbps,
                args.variation_gbps,
                args.seed_start,
                args.episodes,
                args.horizon,
            )
            if spec.rssi:
                prefix = baseline.split("_")[0].upper()
                low_threshold = thresholds.get(
                    f"{prefix}_RSSI_THRESHOLD_LO",
                    thresholds.get("RSSI_THRESHOLD_LO"),
                )
                high_threshold = thresholds.get(
                    f"{prefix}_RSSI_THRESHOLD_HI",
                    thresholds.get("RSSI_THRESHOLD_HI"),
                )
                if low_threshold is None or high_threshold is None:
                    raise ValueError(
                        f"Missing thresholds for {baseline}"
                    )
                command.extend([
                    "--baseline_policy", "rssi",
                    "--rssi_cluster_size",
                    str(args.rssi_cluster_size),
                    "--rssi_threshold_lo", str(low_threshold),
                    "--rssi_threshold_hi", str(high_threshold),
                    "--cluster_policy", spec.cluster_policy,
                    "--policy_label", baseline,
                ])
            else:
                command.extend([
                    "--baseline_policy", "fixed",
                    "--fixed_action", spec.fixed_action,
                    "--cluster_policy", spec.cluster_policy,
                    "--policy_label", baseline,
                ])
            existing_candidates = (
                sorted(eval_dir.glob("*.json"))
                if eval_dir.exists() else []
            )
            if args.skip_existing and existing_candidates:
                result_path = existing_candidates[-1]
            else:
                execute_evaluation(
                    command,
                    project_root,
                    eval_dir / "evaluate.log",
                    args.dry_run,
                )
                if args.dry_run:
                    continue
                result_path = find_result_json(eval_dir)
            payload = json.loads(
                result_path.read_text(encoding="utf-8")
            )
            raw_rows.append({
                "strategy": baseline,
                "strategy_label": BASELINE_LABELS[baseline],
                "capacity_gbps": capacity_gbps,
                "backhaul_lo_mbps": low / 1e6,
                "backhaul_hi_mbps": high / 1e6,
                "model_seed": None,
                "sum_rate_mbps": payload["sum_rate_mbps"],
                "per_ue_rate_p05_mbps": payload[
                    "per_ue_rate_p05_mbps"
                ],
                "qos_violation_rate": payload[
                    "qos_violation_rate"
                ],
                "overload_ratio": payload["overload_ratio"],
                "fronthaul_queue_delay_ms": payload[
                    "fronthaul_queue_delay_ms"
                ],
                "cluster_change_rate": payload[
                    "cluster_change_rate"
                ],
                "evaluation_json": str(result_path),
            })
            result_paths.append(result_path)

    if args.dry_run:
        print("\nDRY RUN: no summary or plots generated.")
        return

    aggregate = aggregate_rows(raw_rows)
    write_csv(output_dir / "raw_results.csv", raw_rows)
    write_csv(
        output_dir / "aggregate_results.csv", aggregate
    )
    digests = {}
    for path in result_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        capacity = int(round(
            (
                float(payload["config"]["backhaul_lo"])
                + float(payload["config"]["backhaul_hi"])
            ) / 2e9 * 1000
        ))
        digest = tuple(
            payload["evaluation_protocol"]["initial_state_digests"]
        )
        digests.setdefault(capacity, set()).add(digest)
    manifest = {
        "analysis": "zero-shot average backhaul-capacity sweep",
        "trained_average_backhaul_gbps": (
            args.reference_capacity_gbps
        ),
        "capacity_variation_gbps": args.variation_gbps,
        "capacities_gbps": capacities,
        "families": families,
        "baselines": baselines,
        "seed_start": args.seed_start,
        "episodes": args.episodes,
        "horizon": args.horizon,
        "rssi_thresholds_env": (
            str(threshold_path) if threshold_path else None
        ),
        "scenario_digests_consistent": {
            str(key): len(value) == 1
            for key, value in sorted(digests.items())
        },
        "raw_result_rows": len(raw_rows),
    }
    (output_dir / "sweep_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_all_capacities(
        aggregate,
        capacities,
        output_dir / "backhaul_capacity_all.png",
    )
    plot_main_comparison(
        aggregate,
        capacities,
        output_dir / "backhaul_capacity_main.png",
    )
    plot_paper_figure5_style(
        aggregate,
        capacities,
        output_dir / "backhaul_figure5_style.png",
    )
    print(f"OUTPUT_DIR={output_dir}")
    print(f"ALL_FIGURE={output_dir / 'backhaul_capacity_all.png'}")
    print(f"MAIN_FIGURE={output_dir / 'backhaul_capacity_main.png'}")
    print(
        "FIGURE5_FIGURE="
        f"{output_dir / 'backhaul_figure5_style.png'}"
    )


if __name__ == "__main__":
    main()
