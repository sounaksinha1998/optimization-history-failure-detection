"""Tests for Phase 0 baseline instrumentation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.common.datasets import NoisyMNISTBundle, NoisyMNISTConfig
from research.common.metrics import PHASE0_COLUMNS, records_from_logits, softmax_probs
from research.common.plotting import plot_phase0
from research.phase0_baseline.run import Phase0Config, run_phase0


def test_records_from_logits_known_values():
    logits = np.array([[10.0, 0.0], [0.0, 10.0]], dtype=np.float64)
    labels = np.array([0, 0], dtype=np.int64)
    ids = np.array(["val_00000", "val_00001"])
    df = records_from_logits(logits, labels, ids, seed=42, epoch=1, step=7, split="val")
    assert list(df.columns) == PHASE0_COLUMNS
    assert df.loc[0, "prediction"] == 0
    assert df.loc[0, "correct"] == 1
    assert df.loc[1, "prediction"] == 1
    assert df.loc[1, "correct"] == 0
    assert df.loc[0, "confidence"] > 0.99
    assert df.loc[0, "loss"] < 0.01
    assert df.loc[1, "loss"] > 1.0
    probs = softmax_probs(logits)
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=-1)
    np.testing.assert_allclose(df["predictive_entropy"].to_numpy(), entropy, rtol=1e-6)


def _tiny_bundle() -> NoisyMNISTBundle:
    rng = np.random.default_rng(0)
    n_train, n_val, n_test = 32, 16, 16
    x_train = rng.random((n_train, 28, 28, 1), dtype=np.float32)
    x_val = rng.random((n_val, 28, 28, 1), dtype=np.float32)
    x_test = rng.random((n_test, 28, 28, 1), dtype=np.float32)
    y_train = rng.integers(0, 10, size=n_train, dtype=np.int32)
    y_val = rng.integers(0, 10, size=n_val, dtype=np.int32)
    y_test = rng.integers(0, 10, size=n_test, dtype=np.int32)
    return NoisyMNISTBundle(
        name="tiny",
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        sample_ids={
            "train": np.array([f"train_{i:05d}" for i in range(n_train)], dtype=object),
            "val": np.array([f"val_{i:05d}" for i in range(n_val)], dtype=object),
            "test": np.array([f"test_{i:05d}" for i in range(n_test)], dtype=object),
        },
        metadata=pd.DataFrame(),
        config=NoisyMNISTConfig(variant="mnist_clean"),
        sigma=0.0,
        mnist_indices={
            "train": np.arange(n_train, dtype=np.int32),
            "val": np.arange(n_val, dtype=np.int32),
            "test": np.arange(n_test, dtype=np.int32),
        },
    )


def test_phase0_run_writes_artifacts(tmp_path: Path):
    pytest.importorskip("jax")
    cfg = Phase0Config(
        seeds=(0, 1),
        epochs=2,
        batch_size=8,
        learning_rate=1e-3,
        output_dir=tmp_path,
        data_dir=tmp_path,
    )
    run_phase0(_tiny_bundle(), cfg)

    baseline = tmp_path / "phase0_baseline.csv"
    per_sample = tmp_path / "per_sample.csv"
    metrics = tmp_path / "metrics.csv"
    assert baseline.exists()
    assert per_sample.exists()
    assert metrics.exists()
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "README.md").exists()
    for name in (
        "accuracy_vs_epoch.png",
        "mean_loss_vs_epoch.png",
        "confidence_distribution.png",
        "entropy_distribution.png",
    ):
        assert (tmp_path / "plots" / name).exists()

    df = pd.read_csv(baseline)
    assert list(df.columns) == PHASE0_COLUMNS
    assert set(df["seed"].unique()) == {0, 1}
    assert set(df["split"].unique()) == {"val", "test"}
    # 2 seeds × 2 epochs × (16 val + 16 test)
    assert len(df) == 2 * 2 * (16 + 16)
    assert df["loss"].notna().all()
    assert df["confidence"].between(0.0, 1.0).all()
    assert (df["predictive_entropy"] >= 0).all()

    cfg_json = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert cfg_json["optimizer"] == "adam"
    assert len(cfg_json["runs"]) == 2


def test_plot_phase0_from_tables(tmp_path: Path):
    per_sample = pd.DataFrame(
        {
            "seed": [0, 0, 0, 0],
            "epoch": [0, 0, 1, 1],
            "step": [1, 1, 2, 2],
            "split": ["val", "val", "val", "val"],
            "sample_id": ["a", "b", "a", "b"],
            "label": [0, 1, 0, 1],
            "prediction": [0, 0, 0, 1],
            "correct": [1, 0, 1, 1],
            "loss": [0.1, 1.2, 0.05, 0.2],
            "confidence": [0.9, 0.4, 0.95, 0.8],
            "predictive_entropy": [0.3, 1.1, 0.2, 0.5],
        }
    )
    metrics = pd.DataFrame(
        {
            "seed": [0, 0],
            "epoch": [0, 1],
            "split": ["val", "val"],
            "accuracy": [0.5, 1.0],
            "mean_loss": [0.65, 0.125],
        }
    )
    paths = plot_phase0(metrics, per_sample, tmp_path)
    assert all(p.exists() for p in paths.values())
    assert set(paths) == {
        "accuracy_vs_epoch",
        "mean_loss_vs_epoch",
        "confidence_distribution",
        "entropy_distribution",
    }
