"""Tests for Phase 5 OOD / distribution-shift analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.common.ood_detection import (
    decision_gate,
    evaluate_shift_scorers,
    ood_label_array,
    run_phase5_analysis,
    score_ood_auroc_auprc,
)
from research.common.shift_datasets import rotate_images, translate_images


def _synthetic_ood_df(n_id: int = 200, n_ood: int = 200, shift: str = "rotated") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    id_entropy = rng.uniform(0.1, 1.0, size=n_id)
    ood_entropy = rng.uniform(1.2, 2.5, size=n_ood)
    id_novelty = 0.5 * id_entropy + rng.normal(0, 0.05, size=n_id)
    ood_novelty = 0.5 * ood_entropy + rng.normal(0, 0.05, size=n_ood)
    id_disagreement = rng.uniform(0.0, 0.2, size=n_id)
    ood_disagreement = rng.uniform(0.3, 0.8, size=n_ood)

    id_df = pd.DataFrame(
        {
            "seed": np.full(n_id, 42),
            "shift": "mnist",
            "domain": "id",
            "is_ood": 0,
            "predictive_entropy": id_entropy,
            "memory_novelty": id_novelty,
            "memory_disagreement": id_disagreement,
        }
    )
    ood_df = pd.DataFrame(
        {
            "seed": np.full(n_ood, 42),
            "shift": shift,
            "domain": "ood",
            "is_ood": 1,
            "predictive_entropy": ood_entropy,
            "memory_novelty": ood_novelty,
            "memory_disagreement": ood_disagreement,
        }
    )
    return pd.concat([id_df, ood_df], ignore_index=True)


def test_ood_label_array():
    labels = ood_label_array(3, 2)
    assert list(labels) == [0, 0, 0, 1, 1]


def test_score_ood_auroc_detects_shift():
    rng = np.random.default_rng(1)
    id_scores = rng.uniform(0.0, 0.5, size=100)
    ood_scores = rng.uniform(1.0, 2.0, size=100)
    labels = ood_label_array(len(id_scores), len(ood_scores))
    scores = np.concatenate([id_scores, ood_scores])
    metrics = score_ood_auroc_auprc(labels, scores)
    assert metrics["auroc"] > 0.95


def test_evaluate_shift_scorers_runs():
    df = _synthetic_ood_df()
    id_df = df[df["shift"] == "mnist"]
    ood_df = df[df["shift"] == "rotated"]
    metrics = evaluate_shift_scorers(id_df, ood_df)
    assert len(metrics) == 3
    assert metrics["auroc"].gt(0.7).all()


def test_run_phase5_writes_artifacts(tmp_path: Path):
    df = _synthetic_ood_df()
    result = run_phase5_analysis(df, tmp_path)
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "decision_gate.csv").exists()
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "plots" / "roc_curves_rotated.png").exists()
    assert result["decision_gate"]["passed"] is True


def test_decision_gate_fail_when_no_separation():
    metrics = pd.DataFrame(
        [
            {"shift": "rotated", "scorer": "predictive_entropy", "auroc": 0.52, "auprc": 0.5, "n": 100},
            {"shift": "rotated", "scorer": "memory_novelty", "auroc": 0.51, "auprc": 0.5, "n": 100},
        ]
    )
    gate = decision_gate(metrics, min_auroc=0.55)
    assert gate["passed"] is False


def test_rotate_and_translate_preserve_shape():
    x = np.random.rand(4, 28, 28, 1).astype(np.float32)
    assert rotate_images(x, 30.0).shape == x.shape
    assert translate_images(x, 2, -3).shape == x.shape
