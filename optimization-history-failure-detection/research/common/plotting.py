"""Plot helpers for multi-timescale memory verification phases."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _latest_eval_split(per_sample: pd.DataFrame) -> pd.DataFrame:
    if "split" not in per_sample.columns:
        return per_sample
    preferred = "val" if (per_sample["split"] == "val").any() else per_sample["split"].iloc[-1]
    subset = per_sample[per_sample["split"] == preferred]
    last_epoch = int(subset["epoch"].max())
    return subset[subset["epoch"] == last_epoch]


def plot_phase0(
    metrics: pd.DataFrame,
    per_sample: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Path]:
    """Write Phase-0 diagnostic plots. Returns path map."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    eval_metrics = metrics[metrics["split"] == "val"] if "split" in metrics.columns else metrics

    fig, ax = plt.subplots(figsize=(6, 4))
    for seed, g in eval_metrics.groupby("seed"):
        g = g.sort_values("epoch")
        ax.plot(g["epoch"], g["accuracy"], marker="o", label=f"seed {seed}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs epoch")
    if eval_metrics["seed"].nunique() > 1:
        ax.legend()
    fig.tight_layout()
    paths["accuracy_vs_epoch"] = out_dir / "accuracy_vs_epoch.png"
    fig.savefig(paths["accuracy_vs_epoch"], dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for seed, g in eval_metrics.groupby("seed"):
        g = g.sort_values("epoch")
        ax.plot(g["epoch"], g["mean_loss"], marker="o", label=f"seed {seed}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean loss")
    ax.set_title("Mean loss vs epoch")
    if eval_metrics["seed"].nunique() > 1:
        ax.legend()
    fig.tight_layout()
    paths["mean_loss_vs_epoch"] = out_dir / "mean_loss_vs_epoch.png"
    fig.savefig(paths["mean_loss_vs_epoch"], dpi=120)
    plt.close(fig)

    latest = _latest_eval_split(per_sample)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(latest["confidence"], bins=30, color="steelblue", edgecolor="white")
    ax.set_xlabel("confidence")
    ax.set_ylabel("count")
    ax.set_title("Confidence distribution")
    fig.tight_layout()
    paths["confidence_distribution"] = out_dir / "confidence_distribution.png"
    fig.savefig(paths["confidence_distribution"], dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(latest["predictive_entropy"], bins=30, color="darkorange", edgecolor="white")
    ax.set_xlabel("predictive entropy")
    ax.set_ylabel("count")
    ax.set_title("Predictive entropy distribution")
    fig.tight_layout()
    paths["entropy_distribution"] = out_dir / "entropy_distribution.png"
    fig.savefig(paths["entropy_distribution"], dpi=120)
    plt.close(fig)

    return paths


def plot_memory_phase(
    metrics: pd.DataFrame,
    per_sample: pd.DataFrame,
    out_dir: Path,
    *,
    phase_level: int,
) -> dict[str, Path]:
    """Diagnostic plots for phases 1–3."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = plot_phase0(metrics, per_sample, out_dir)

    eval_metrics = metrics[metrics["split"] == "val"] if "split" in metrics.columns else metrics
    if "mem_alpha_norm" in eval_metrics.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        for seed, g in eval_metrics.groupby("seed"):
            g = g.sort_values("epoch")
            ax.plot(g["epoch"], g["mem_alpha_norm"], marker="o", label=f"seed {seed}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("‖A‖")
        ax.set_title("Memory alpha norm vs epoch")
        if eval_metrics["seed"].nunique() > 1:
            ax.legend()
        fig.tight_layout()
        paths["mem_alpha_norm_vs_epoch"] = out_dir / "mem_alpha_norm_vs_epoch.png"
        fig.savefig(paths["mem_alpha_norm_vs_epoch"], dpi=120)
        plt.close(fig)

    latest = _latest_eval_split(per_sample)
    if phase_level >= 2 and "c_1" in latest.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(latest["c_1"].dropna(), bins=30, color="seagreen", edgecolor="white", alpha=0.85)
        ax.set_xlabel("c_1 (fast-scale alignment)")
        ax.set_ylabel("count")
        ax.set_title("Alignment c_1 distribution")
        fig.tight_layout()
        paths["alignment_c1_distribution"] = out_dir / "alignment_c1_distribution.png"
        fig.savefig(paths["alignment_c1_distribution"], dpi=120)
        plt.close(fig)

    if phase_level >= 3 and "memory_novelty" in latest.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(latest["memory_novelty"].dropna(), bins=30, color="purple", edgecolor="white", alpha=0.85)
        ax.set_xlabel("memory novelty")
        ax.set_ylabel("count")
        ax.set_title("Memory novelty distribution")
        fig.tight_layout()
        paths["memory_novelty_distribution"] = out_dir / "memory_novelty_distribution.png"
        fig.savefig(paths["memory_novelty_distribution"], dpi=120)
        plt.close(fig)

    return paths
