"""Tests for Phase 4 error-prediction analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.common.error_prediction import (
    add_error_target,
    decision_gate,
    evaluate_all_single_features,
    evaluate_combined_models,
    future_error_correlation,
    run_phase4_analysis,
)


def _synthetic_phase3_df(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    entropy = rng.uniform(0.0, 2.5, size=n)
    novelty = 0.6 * entropy + rng.normal(0.0, 0.2, size=n)
    disagreement = 0.5 * entropy + rng.normal(0.0, 0.25, size=n)
    msa_entropy = 0.4 * entropy + rng.normal(0.0, 0.2, size=n)
    error_prob = 1.0 / (1.0 + np.exp(-(entropy - 1.0)))
    error = (rng.random(n) < error_prob).astype(int)
    correct = 1 - error
    split = np.array(["val"] * (n // 2) + ["test"] * (n - n // 2))
    sample_ids = np.array([f"s_{i % 40:03d}" for i in range(n)], dtype=object)
    epochs = np.tile(np.arange(5), n // 5 + 1)[:n]
    return pd.DataFrame(
        {
            "seed": np.full(n, 42),
            "epoch": epochs,
            "step": np.arange(n),
            "split": split,
            "sample_id": sample_ids,
            "label": rng.integers(0, 10, size=n),
            "prediction": rng.integers(0, 10, size=n),
            "correct": correct,
            "loss": rng.uniform(0.0, 2.0, size=n),
            "confidence": rng.uniform(0.1, 1.0, size=n),
            "predictive_entropy": entropy,
            "memory_novelty": novelty,
            "memory_disagreement": disagreement,
            "msa_entropy": msa_entropy,
        }
    )


def test_add_error_target():
    df = pd.DataFrame({"correct": [1, 0, 1]})
    out = add_error_target(df)
    assert list(out["error"]) == [0, 1, 0]


def test_single_feature_metrics_improve_with_signal():
    df = add_error_target(_synthetic_phase3_df())
    metrics = evaluate_all_single_features(df)
    test_entropy = metrics[
        (metrics["split"] == "test") & (metrics["feature"] == "predictive_entropy")
    ].iloc[0]["auroc"]
    test_novelty = metrics[
        (metrics["split"] == "test") & (metrics["feature"] == "memory_novelty")
    ].iloc[0]["auroc"]
    assert test_entropy > 0.55
    assert test_novelty > 0.55


def test_combined_model_runs_and_returns_metrics():
    df = add_error_target(_synthetic_phase3_df(n=2000, seed=1))
    combined = evaluate_combined_models(df)
    test = combined[combined["split"] == "test"]
    entropy_auroc = float(test[test["model"] == "entropy_only"]["auroc"].iloc[0])
    combined_auroc = float(test[test["model"] == "combined"]["auroc"].iloc[0])
    assert entropy_auroc > 0.55
    assert combined_auroc > 0.55


def test_future_error_correlation_runs():
    df = add_error_target(_synthetic_phase3_df())
    corr = future_error_correlation(df)
    assert len(corr) == 4
    assert corr["n_pairs"].gt(0).all()


def test_run_phase4_writes_artifacts(tmp_path: Path):
    df = _synthetic_phase3_df()
    result = run_phase4_analysis(df, tmp_path)
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "metrics_combined.csv").exists()
    assert (tmp_path / "decision_gate.csv").exists()
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "plots" / "error_probability_vs_uncertainty_test.png").exists()
    assert "passed" in result["decision_gate"]


def test_decision_gate_fail_when_no_signal():
    single = pd.DataFrame(
        [
            {"split": "test", "feature": "predictive_entropy", "auroc": 0.80, "auprc": 0.5, "n": 100},
            {"split": "test", "feature": "memory_novelty", "auroc": 0.79, "auprc": 0.4, "n": 100},
            {"split": "test", "feature": "memory_disagreement", "auroc": 0.78, "auprc": 0.4, "n": 100},
            {"split": "test", "feature": "msa_entropy", "auroc": 0.77, "auprc": 0.4, "n": 100},
        ]
    )
    combined = pd.DataFrame(
        [
            {"split": "test", "model": "entropy_only", "auroc": 0.80, "auprc": 0.5, "n": 100},
            {"split": "test", "model": "combined", "auroc": 0.801, "auprc": 0.5, "n": 100},
        ]
    )
    gate = decision_gate(single, combined, auroc_margin=0.005)
    assert gate["passed"] is False
