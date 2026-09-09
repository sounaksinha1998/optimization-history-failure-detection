"""Shared orchestration for research phases 1–3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from research.common.memory_training import (
    DEFAULT_SEEDS,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    TRAIN_SUBSET,
    LONG_TERM_DEPTH,
    MEMORY_TAU,
    ATTENTION_TAU,
    MemoryPhaseConfig,
    train_one_seed,
)
from research.common.metrics import phase_columns_for_level
from research.common.plotting import plot_memory_phase
from research.phase0_baseline.run import load_clean_mnist_bundle

PHASE_ARTIFACTS = {
    1: "phase1_memory.csv",
    2: "phase2_msa.csv",
    3: "phase3_signals.csv",
}

PHASE_DIRS = {
    1: "phase1_memory",
    2: "phase2_msa",
    3: "phase3_signals",
}

PHASE_TITLES = {
    1: "Phase 1 — V5B memory as observation only",
    2: "Phase 2 — MSA activation",
    3: "Phase 3 — Memory-derived signals",
}


def write_phase_readme(
    path: Path,
    *,
    phase_level: int,
    cfg: MemoryPhaseConfig,
    summaries: list[dict[str, Any]],
    metrics: pd.DataFrame,
) -> None:
    val = metrics[metrics["split"] == "val"]
    last_epoch = int(val["epoch"].max()) if len(val) else -1
    last = val[val["epoch"] == last_epoch] if last_epoch >= 0 else val
    mean_acc = float(last["accuracy"].mean()) if len(last) else float("nan")
    mean_loss = float(last["mean_loss"].mean()) if len(last) else float("nan")
    seeds = ", ".join(str(s) for s in cfg.seeds)
    artifact = PHASE_ARTIFACTS[phase_level]
    title = PHASE_TITLES[phase_level]

    hypotheses = {
        1: (
            "Multi-timescale V5B memories L^(j) can be maintained alongside Adam updates",
            "without modifying the optimizer step (observation only).",
        ),
        2: (
            "Per-example gradient alignments c_j = ⟨ĝ, L̂^(j)⟩ and attention weights",
            f"α_j = softmax(c_j / τ) with τ={cfg.attention_tau} can be computed at evaluation time.",
        ),
        3: (
            "Agreement, novelty, cross-timescale disagreement, and MSA entropy can be",
            "derived per sample from alignments and attention weights.",
        ),
    }
    decisions = {
        1: "V5B memory instrumentation is in place without changing Adam. Proceed to Phase 2 (MSA activation).",
        2: "MSA alignments and attention weights are logged. Proceed to Phase 3 (memory-derived signals).",
        3: "Per-sample memory signals are available. Proceed to Phase 4 (error prediction test).",
    }
    next_phases = {
        1: "Phase 2 — compute normalized gradient/memory similarities and softmax attention.",
        2: "Phase 3 — implement agreement, novelty, disagreement, and MSA entropy per sample.",
        3: "Phase 4 — test whether memory signals predict model errors (AUROC / AUPRC).",
    }

    hyp_lines = hypotheses[phase_level]
    lines = [
        f"# {title}",
        "",
        "## Hypothesis",
        "",
        hyp_lines[0],
    ]
    if len(hyp_lines) > 1:
        lines.append(hyp_lines[1])
    lines.extend(
        [
            "",
            "## Experiment",
            "",
            f"- Dataset: MNIST (`{cfg.variant}`)",
            f"- Optimizer: `research_nrm_v2_observe` (Adam updates, memory observation only)",
            f"- Long-term depth K={cfg.long_term_depth}, memory τ={cfg.memory_tau}",
            f"- Seeds: {seeds}",
            f"- Epochs: {cfg.epochs}, batch size: {cfg.batch_size}, train subset: {cfg.train_subset}",
            f"- Primary artifact: `{artifact}`",
            "",
            "## Result",
            "",
            f"- Runs completed: {len(summaries)}",
            f"- Mean val accuracy (last epoch): {mean_acc:.4f}",
            f"- Mean val loss (last epoch): {mean_loss:.4f}",
            "- Artifacts: primary CSV, `per_sample.csv`, `metrics.csv`, `plots/`",
            "",
            "## Decision",
            "",
            decisions[phase_level],
            "",
            "## Next phase",
            "",
            next_phases[phase_level],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_memory_phase(bundle, cfg: MemoryPhaseConfig) -> dict[str, Any]:
    level = cfg.phase_level
    output_dir = Path(cfg.output_dir)
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_sample: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    for seed in cfg.seeds:
        print(f"=== phase{level} nrm_v2_observe seed={seed} ===")
        per_sample, metrics, summary = train_one_seed(bundle, seed=seed, cfg=cfg)
        all_sample.append(per_sample)
        all_metrics.append(metrics)
        summaries.append(summary)
        print(f"  test_acc={summary['test_acc']:.4f}, time={summary['train_time_s']:.1f}s")

    columns = phase_columns_for_level(level, long_term_depth=cfg.long_term_depth)
    per_sample_df = pd.concat(all_sample, ignore_index=True)
    for col in columns:
        if col not in per_sample_df.columns:
            per_sample_df[col] = float("nan")
    per_sample_df = per_sample_df[columns]
    metrics_df = pd.concat(all_metrics, ignore_index=True)

    artifact_name = PHASE_ARTIFACTS[level]
    per_sample_df.to_csv(output_dir / "per_sample.csv", index=False)
    per_sample_df.to_csv(output_dir / artifact_name, index=False)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    plot_memory_phase(metrics_df, per_sample_df, plots_dir, phase_level=level)

    config = {
        "phase": level,
        "seeds": list(cfg.seeds),
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "variant": cfg.variant,
        "train_subset": cfg.train_subset,
        "long_term_depth": cfg.long_term_depth,
        "memory_tau": cfg.memory_tau,
        "attention_tau": cfg.attention_tau,
        "data_dir": str(cfg.data_dir),
        "output_dir": str(cfg.output_dir),
        "optimizer": "nrm_v2_observe",
        "runs": summaries,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_phase_readme(output_dir / "README.md", phase_level=level, cfg=cfg, summaries=summaries, metrics=metrics_df)
    return config
