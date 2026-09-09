"""Phase-0 predictive statistics: loss, confidence, and predictive entropy."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

PHASE0_COLUMNS = [
    "seed",
    "epoch",
    "step",
    "split",
    "sample_id",
    "label",
    "prediction",
    "correct",
    "loss",
    "confidence",
    "predictive_entropy",
]

PHASE1_MEMORY_COLUMNS = [
    "optimizer_step",
    "mem_gamma_norm",
    "mem_beta_norm",
    "mem_alpha_norm",
    "mem_L1_norm",
    "mem_L2_norm",
    "mem_L3_norm",
]

PHASE2_MSA_COLUMNS = [
    "c_1",
    "c_2",
    "c_3",
    "alpha_1",
    "alpha_2",
    "alpha_3",
]

PHASE3_SIGNAL_COLUMNS = [
    "memory_agreement",
    "memory_novelty",
    "memory_disagreement",
    "msa_entropy",
]

PHASE1_COLUMNS = PHASE0_COLUMNS + PHASE1_MEMORY_COLUMNS
PHASE2_COLUMNS = PHASE1_COLUMNS + PHASE2_MSA_COLUMNS
PHASE3_COLUMNS = PHASE2_COLUMNS + PHASE3_SIGNAL_COLUMNS


def phase_memory_columns(depth: int = 3) -> list[str]:
    base = ["optimizer_step", "mem_gamma_norm", "mem_beta_norm", "mem_alpha_norm"]
    return base + [f"mem_L{j}_norm" for j in range(1, depth + 1)]


def phase_msa_columns(depth: int = 3) -> list[str]:
    return [f"c_{j}" for j in range(1, depth + 1)] + [f"alpha_{j}" for j in range(1, depth + 1)]


def phase_columns_for_level(level: int, *, long_term_depth: int = 3) -> list[str]:
    """Column order for research phase 1, 2, or 3."""
    if level < 1 or level > 3:
        raise ValueError(f"level must be 1, 2, or 3, got {level}")
    cols = list(PHASE0_COLUMNS) + phase_memory_columns(long_term_depth)
    if level >= 2:
        cols.extend(phase_msa_columns(long_term_depth))
    if level >= 3:
        cols.extend(PHASE3_SIGNAL_COLUMNS)
    return cols


def softmax_probs(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax over the class axis."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def cross_entropy_from_probs(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-example CE(p, y) with integer labels."""
    labels = np.asarray(labels, dtype=np.int64)
    gathered = np.take_along_axis(probs, labels[:, None], axis=1)[:, 0]
    return -np.log(np.clip(gathered, 1e-12, 1.0))


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    """H(p) = -sum_y p(y) log p(y)."""
    safe = np.clip(probs, 1e-12, 1.0)
    return -np.sum(safe * np.log(safe), axis=-1)


def records_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    *,
    seed: int,
    epoch: int,
    step: int,
    split: str,
) -> pd.DataFrame:
    """Build one Phase-0 row per evaluation example from classifier logits."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    sample_ids = np.asarray(sample_ids)
    if logits.ndim != 2:
        raise ValueError(f"logits must be (N, C), got {logits.shape}")
    if len(logits) != len(labels) or len(logits) != len(sample_ids):
        raise ValueError("logits, labels, and sample_ids must have the same length")

    probs = softmax_probs(logits)
    prediction = np.argmax(probs, axis=-1).astype(np.int64)
    labels_i = labels.astype(np.int64)
    return pd.DataFrame(
        {
            "seed": np.full(len(logits), int(seed), dtype=np.int64),
            "epoch": np.full(len(logits), int(epoch), dtype=np.int64),
            "step": np.full(len(logits), int(step), dtype=np.int64),
            "split": np.full(len(logits), split, dtype=object),
            "sample_id": np.array([str(s) for s in sample_ids], dtype=object),
            "label": labels_i,
            "prediction": prediction,
            "correct": (prediction == labels_i).astype(np.int64),
            "loss": cross_entropy_from_probs(probs, labels_i),
            "confidence": np.max(probs, axis=-1),
            "predictive_entropy": predictive_entropy(probs),
        }
    )[PHASE0_COLUMNS]


def summarize_epoch(per_sample: pd.DataFrame, *, train_loss: float | None = None) -> list[dict[str, Any]]:
    """Accuracy and mean loss per (seed, epoch, split) from per-sample logs."""
    rows: list[dict[str, Any]] = []
    grouped = per_sample.groupby(["seed", "epoch", "split"], sort=True)
    for (seed, epoch, split), g in grouped:
        row: dict[str, Any] = {
            "seed": int(seed),
            "epoch": int(epoch),
            "split": str(split),
            "n": int(len(g)),
            "accuracy": float(g["correct"].mean()),
            "mean_loss": float(g["loss"].mean()),
            "mean_confidence": float(g["confidence"].mean()),
            "mean_predictive_entropy": float(g["predictive_entropy"].mean()),
        }
        if train_loss is not None and split == "val":
            row["train_loss"] = float(train_loss)
        rows.append(row)
    return rows
