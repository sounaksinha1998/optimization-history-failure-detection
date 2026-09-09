"""Tests for memory novelty complementarity analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.common.complementarity import (
    ComplementarityConfig,
    assign_quadrant,
    build_per_example_table,
    compare_quadrants,
    fit_thresholds,
    quadrant_error_rates,
    run_complementarity_analysis,
)


def _synthetic_df(n_cal: int = 400, n_test: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for split, n in (("val", n_cal), ("test", n_test)):
        h = rng.uniform(0.0, 2.5, size=n)
        n_sig = 0.5 * h + rng.normal(0, 0.3, size=n)
        error_prob = 1 / (1 + np.exp(-(0.8 * h + 1.5 * n_sig - 1.5)))
        error = (rng.random(n) < error_prob).astype(int)
        correct = 1 - error
        for i in range(n):
            rows.append(
                {
                    "seed": 42,
                    "epoch": 4,
                    "step": i,
                    "split": split,
                    "sample_id": f"{split}_{i:05d}",
                    "label": int(rng.integers(0, 10)),
                    "prediction": int(rng.integers(0, 10)),
                    "correct": int(correct[i]),
                    "predictive_entropy": float(h[i]),
                    "memory_novelty": float(n_sig[i]),
                }
            )
    return pd.DataFrame(rows)


def test_build_per_example_table():
    df = build_per_example_table(_synthetic_df())
    assert "H" in df.columns
    assert "N" in df.columns
    assert set(df["error"]) <= {0, 1}


def test_assign_quadrants():
    df = build_per_example_table(_synthetic_df())
    cal = df[df["split"] == "val"]
    thresholds = fit_thresholds(cal)
    test = df[df["split"] == "test"].copy()
    test["quadrant"] = assign_quadrant(test, thresholds)
    assert set(test["quadrant"].dropna()) <= {"A", "B", "C", "D"}


def test_quadrant_error_rates():
    df = build_per_example_table(_synthetic_df())
    cal = df[df["split"] == "val"]
    test = df[df["split"] == "test"].copy()
    test["quadrant"] = assign_quadrant(test, fit_thresholds(cal))
    rates = quadrant_error_rates(test, n_bootstrap=50, bootstrap_seed=0)
    assert len(rates) == 4
    assert rates["error_rate"].between(0, 1).all()


def test_compare_quadrants():
    df = build_per_example_table(_synthetic_df())
    cal = df[df["split"] == "val"]
    test = df[df["split"] == "test"].copy()
    test["quadrant"] = assign_quadrant(test, fit_thresholds(cal))
    rates = quadrant_error_rates(test, n_bootstrap=50)
    cmp = compare_quadrants(rates, "A", "B", test, n_bootstrap=50)
    assert "delta" in cmp


def test_run_complementarity_writes_artifacts(tmp_path: Path):
    cfg = ComplementarityConfig(
        output_dir=tmp_path,
        n_bootstrap=50,
        save_artifacts=True,
        show_plots=False,
    )
    result = run_complementarity_analysis(_synthetic_df(), cfg)
    assert (tmp_path / "per_example.csv").exists()
    assert (tmp_path / "quadrant_error_rates.csv").exists()
    assert (tmp_path / "logistic_models.csv").exists()
    assert (tmp_path / "decision_gate.csv").exists()
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "plots" / "reliability_map_2d.png").exists()
    assert "passed" in result["decision_gate"]
