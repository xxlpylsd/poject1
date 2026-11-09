# -*- coding: utf-8 -*-
"""Plot reference-paper-style training reward convergence curves."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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


def flatten_reward_array(values: np.ndarray) -> np.ndarray:
    """Average environments, agents and singleton reward axes per step."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        raise ValueError("Reward array must contain at least one step")
    return array.reshape(array.shape[0], -1).mean(axis=1)


def group_reward_episodes(
    step_rewards: np.ndarray,
    episode_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-episode mean and within-episode standard deviation."""
    if episode_length <= 0:
        raise ValueError("episode_length must be positive")
    values = np.asarray(step_rewards, dtype=float).reshape(-1)
    count = values.size // episode_length
    if count < 1:
        raise ValueError(
            "Reward history is shorter than one configured episode"
        )
    blocks = values[:count * episode_length].reshape(
        count, episode_length
    )
    means = np.mean(blocks, axis=1)
    stds = np.std(blocks, axis=1, ddof=1) if episode_length > 1 else np.zeros(
        count, dtype=float
    )
    return means, stds


def reward_column(reward_mode: str) -> str:
    return {
        "team_rate": "sum_rate_mbps",
        "mean_rate": "team_rate_mbps",
        "mean_se": "mean_ue_se_bps_hz",
        "team_sum": "system_sum_se_bps_hz",
        "paper": "team_rate_mbps",
        "per_ue": "mean_ue_se_bps_hz",
    }.get(str(reward_mode), "sum_rate_mbps")


def load_reward_curve(
    run_dir: Path,
    episode_length: int,
    source: str = "raw",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    """Load raw rewards when available, otherwise use the logged curve."""
    reward_path = run_dir / "reward_proposed.npy"
    raw = None
    if source != "curve" and reward_path.exists():
        raw = np.load(reward_path)
    elif source != "curve":
        reward_path = run_dir / "reward_proposed.mat"
        if reward_path.exists():
            try:
                import scipy.io as sio
            except ImportError as error:
                raise RuntimeError(
                    "Loading reward_proposed.mat requires scipy; "
                    "use reward_proposed.npy or install scipy"
                ) from error
            raw = sio.loadmat(reward_path)["reward_proposed"]
    if source == "raw" and raw is None:
        raise FileNotFoundError(
            "Reference-style convergence requires "
            f"{run_dir / 'reward_proposed.npy'} or reward_proposed.mat"
        )
    if raw is not None:
        step_rewards = flatten_reward_array(raw)
        means, episode_std = group_reward_episodes(
            step_rewards, episode_length
        )
        episodes = np.arange(1, means.size + 1, dtype=int)
        return (
            episodes,
            means,
            episode_std,
            "raw reward_proposed",
            "Reward per decision step",
        )

    config = json.loads(
        (run_dir / "config.json").read_text(encoding="utf-8")
    )
    column = reward_column(config.get("reward_mode", "team_rate"))
    path = run_dir / "logs" / "curve.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No reward_proposed file and no logged curve: {run_dir}"
        )
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty training curve: {path}")
    if column not in rows[0]:
        raise KeyError(
            f"Column {column!r} is unavailable in {path}"
        )
    episodes = np.arange(1, len(rows) + 1, dtype=int)
    means = np.asarray(
        [float(row[column]) for row in rows], dtype=float
    )
    return (
        episodes,
        means,
        np.zeros_like(means),
        "curve.csv fallback",
        f"Logged {column}",
    )


def mean_and_band(
    rows: np.ndarray,
    within_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(rows, axis=0)
    if rows.shape[0] > 1:
        band = (
            1.96
            * np.std(rows, axis=0, ddof=1)
            / np.sqrt(rows.shape[0])
        )
    elif within_std.ndim == 2 and within_std.shape[0] == 1:
        band = 0.95 * within_std[0]
    else:
        band = np.zeros_like(mean)
    return mean, band


def centered_smooth(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return values.copy()
    result = np.empty_like(values, dtype=float)
    for index in range(values.size):
        low = max(0, index - radius)
        high = min(values.size, index + radius + 1)
        result[index] = float(np.mean(values[low:high]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reference-paper-style training reward convergence plot. "
            "Raw per-step rewards are grouped by episode; curve.csv is "
            "used as a clearly-labelled fallback."
        )
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--result-root",
        default="results/IICMSMobility/PureMobility/mappo",
    )
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--families", default="mlp,tf_actor")
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--smoothing-radius", type=int, default=1)
    parser.add_argument(
        "--source",
        choices=("raw", "auto", "curve"),
        default="raw",
        help=(
            "raw requires reward_proposed.npy/.mat; auto falls back to "
            "curve.csv; curve always uses the logged CSV"
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
    data = {}
    sources = set()
    y_label = "Reward per decision step"
    for family in families:
        rows = []
        within = []
        x = None
        for seed in seeds:
            run_dir = latest_run_dir(
                root / f"{args.tag}_{family}_s{seed}",
                requested,
            )
            episodes, means, stds, source, y_label = load_reward_curve(
                run_dir, args.episode_length, args.source
            )
            if x is None:
                x = episodes
            elif not np.array_equal(x, episodes):
                raise ValueError("Reward curves have different lengths")
            rows.append(means)
            within.append(stds)
            sources.add(source)
        data[family] = {
            "x": x,
            "means": np.asarray(rows),
            "within_std": np.asarray(within),
        }

    fig, axis = plt.subplots(figsize=(8.4, 5.2), dpi=180)
    summary = {
        "tag": args.tag,
        "seeds": seeds,
        "episode_length": args.episode_length,
        "source_mode": args.source,
        "data_sources": sorted(sources),
        "families": {},
    }
    for family in families:
        item = data[family]
        for seed, row in zip(seeds, item["means"]):
            axis.plot(
                item["x"],
                centered_smooth(row, args.smoothing_radius),
                color=FAMILY_STYLES[family]["color"],
                alpha=0.18,
                linewidth=1.0,
                label=f"_seed_{family}_{seed}",
            )
        mean, band = mean_and_band(
            item["means"], item["within_std"]
        )
        smooth = centered_smooth(mean, args.smoothing_radius)
        axis.fill_between(
            item["x"],
            mean - band,
            mean + band,
            color=FAMILY_STYLES[family]["color"],
            alpha=0.14,
            linewidth=0,
        )
        axis.plot(
            item["x"],
            smooth,
            color=FAMILY_STYLES[family]["color"],
            linewidth=2.3,
            label=FAMILY_LABELS[family],
        )
        summary["families"][family] = {
            "first": float(smooth[0]),
            "last": float(smooth[-1]),
            "max": float(np.max(smooth)),
            "last_five_mean": float(np.mean(smooth[-5:])),
        }

    axis.set_xlabel("Training episode / logged update")
    axis.set_ylabel(y_label)
    axis.set_title(
        "Training reward convergence"
        if "raw reward_proposed" in sources
        else "Training reward convergence (curve.csv fallback)"
    )
    axis.grid(True, alpha=0.22, linewidth=0.8)
    axis.legend(frameon=False, fontsize=8)
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
