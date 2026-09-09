"""Tests for research phases 1–3 (memory observation, MSA, signals)."""

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
from research.common.metrics import phase_columns_for_level
from research.common.memory_training import MemoryPhaseConfig
from research.common.msa import per_sample_msa_signals, tree_normalized_dot
from research.common.phase_runner import PHASE_ARTIFACTS, run_memory_phase
from research.common.model import init_mlp_params
import jax
import jax.numpy as jnp


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


def test_tree_normalized_dot_unit_vectors():
    pytest.importorskip("jax")
    a = jnp.array([1.0, 0.0, 0.0])
    b = jnp.array([1.0, 0.0, 0.0])
    assert float(tree_normalized_dot(a, b, 1e-8)) == pytest.approx(1.0)
    c = jnp.array([0.0, 1.0, 0.0])
    assert float(tree_normalized_dot(a, c, 1e-8)) == pytest.approx(0.0)


def test_per_sample_msa_signals_properties():
    pytest.importorskip("jax")
    key = jax.random.PRNGKey(0)
    params = init_mlp_params(key)
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    grad = jax.tree_util.tree_map(lambda p: jnp.ones_like(p), params)
    signals = per_sample_msa_signals(grad, (zeros, zeros, zeros), tau=1.0)
    assert signals["memory_agreement"] == pytest.approx(0.0, abs=1e-5)
    assert signals["memory_novelty"] == pytest.approx(1.0, abs=1e-5)
    assert signals["msa_entropy"] >= 0.0
    assert len([k for k in signals if k.startswith("c_")]) == 3
    assert len([k for k in signals if k.startswith("alpha_")]) == 3


@pytest.mark.parametrize("phase_level", [1, 2, 3])
def test_memory_phase_writes_artifacts(tmp_path: Path, phase_level: int):
    pytest.importorskip("jax")
    cfg = MemoryPhaseConfig(
        phase_level=phase_level,  # type: ignore[arg-type]
        seeds=(0,),
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        output_dir=tmp_path / f"phase{phase_level}",
        data_dir=tmp_path,
    )
    run_memory_phase(_tiny_bundle(), cfg)

    artifact = tmp_path / f"phase{phase_level}" / PHASE_ARTIFACTS[phase_level]
    assert artifact.exists()
    assert (tmp_path / f"phase{phase_level}" / "metrics.csv").exists()
    assert (tmp_path / f"phase{phase_level}" / "README.md").exists()

    df = pd.read_csv(artifact)
    expected_cols = phase_columns_for_level(phase_level, long_term_depth=cfg.long_term_depth)
    assert list(df.columns) == expected_cols
    assert df["loss"].notna().all()
    assert df["mem_alpha_norm"].notna().all()

    if phase_level >= 2:
        assert df["c_1"].notna().all()
        assert df["alpha_1"].notna().all()
        row_alpha = df[["alpha_1", "alpha_2", "alpha_3"]].iloc[0].to_numpy(dtype=float)
        np.testing.assert_allclose(row_alpha.sum(), 1.0, rtol=1e-5)

    if phase_level >= 3:
        assert df["memory_agreement"].notna().all()
        assert df["memory_novelty"].notna().all()
        assert df["memory_disagreement"].notna().all()
        assert df["msa_entropy"].notna().all()
        np.testing.assert_allclose(
            df["memory_agreement"] + df["memory_novelty"],
            1.0,
            rtol=1e-5,
            atol=1e-5,
        )

    cfg_json = json.loads((tmp_path / f"phase{phase_level}" / "config.json").read_text(encoding="utf-8"))
    assert cfg_json["phase"] == phase_level
    assert cfg_json["optimizer"] == "nrm_v2_observe"
