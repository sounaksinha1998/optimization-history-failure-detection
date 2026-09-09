"""Tests for label-free deployment pipeline (deployment-pipeline.plan.md)."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from optimizer.associative_memory import (
    AssociativeMemory,
    AssociativeMemoryConfig,
    associative_retrieve,
    init_associative_memory,
    normalize_key,
)
from research.common.deployment_pipeline import (
    compute_cross_level_diagnostics,
    compute_prediction_output,
    compute_representation_and_key,
    deploy_batch,
    deploy_sample,
    join_ground_truth,
    memory_diagnostic_feature_columns,
    offline_evaluate_memory_information,
    query_historical_checkpoints,
    query_memory_levels,
    records_to_dataframe,
    run_deployment_sanity_checks,
    save_deployment_records,
)
from research.common.resnet import init_resnet18_params
from research.common.trajectory_store import MemoryCheckpointStore
from tests.test_associative_memory import _bundle_from_memories, _trained_state


@pytest.fixture
def dims() -> tuple[int, int, int]:
    return 16, 7, 4


@pytest.fixture
def memory_bundle(dims: tuple[int, int, int]):
    d_k, d_v, num_levels = dims
    mem = _trained_state(d_k, d_v, num_levels, steps=20, seed=0)
    states = [mem.state]
    steps = [int(mem.state.step)]
    for _ in range(3):
        mem.update(
            normalize_key(jnp.asarray(np.random.default_rng(1).standard_normal(d_k), dtype=jnp.float32)),
            jnp.asarray(np.random.default_rng(2).standard_normal(d_v), dtype=jnp.float32),
        )
        states.append(mem.state)
        steps.append(int(mem.state.step))
    bundle = _bundle_from_memories(states, steps, sample_every=1)
    return mem.state, bundle


@pytest.fixture
def resnet_memory_bundle():
    """Associative memory with d_k matching ResNet-18 penultimate dim (512)."""
    d_k, d_v, num_levels = 512, 7, 4
    mem = _trained_state(d_k, d_v, num_levels, steps=8, seed=10)
    states = [mem.state]
    steps = [int(mem.state.step)]
    for _ in range(3):
        mem.update(
            normalize_key(jnp.asarray(np.random.default_rng(11).standard_normal(d_k), dtype=jnp.float32)),
            jnp.asarray(np.random.default_rng(12).standard_normal(d_v), dtype=jnp.float32),
        )
        states.append(mem.state)
        steps.append(int(mem.state.step))
    bundle = _bundle_from_memories(states, steps, sample_every=1)
    return mem.state, bundle


@pytest.fixture
def resnet_params_and_image():
    key = jax.random.PRNGKey(0)
    params = init_resnet18_params(key, num_classes=7)
    rng = np.random.default_rng(0)
    x = rng.random((28, 28, 3), dtype=np.float32)
    return params, x


def test_prediction_output_shapes() -> None:
    probs = np.array([0.7, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02], dtype=np.float32)
    out = compute_prediction_output(probs, num_classes=7)
    assert out.predicted_class == 0
    assert out.entropy > 0
    assert 0 <= out.normalized_entropy <= 1.0 + 1e-6


def test_query_memory_levels_manual(dims: tuple[int, int, int]) -> None:
    d_k, d_v, num_levels = dims
    state = init_associative_memory(d_k, d_v, AssociativeMemoryConfig(num_levels=num_levels))
    rng = np.random.default_rng(3)
    for _ in range(5):
        k = normalize_key(jnp.asarray(rng.standard_normal(d_k), dtype=jnp.float32))
        v = jnp.asarray(rng.standard_normal(d_v), dtype=jnp.float32)
        state = __import__(
            "optimizer.associative_memory", fromlist=["associative_update"]
        ).associative_update(state, k, v)
    key = np.asarray(normalize_key(jnp.asarray(rng.standard_normal(d_k), dtype=jnp.float32)), dtype=np.float32)
    levels = query_memory_levels(key, state)
    assert len(levels) == num_levels
    for j, lvl in enumerate(levels, start=1):
        manual = np.asarray(state.matrices[j - 1] @ jnp.asarray(key), dtype=np.float32)
        np.testing.assert_allclose(lvl.response, manual, rtol=1e-5, atol=1e-6)
        assert lvl.magnitude == pytest.approx(float(np.linalg.norm(manual)), rel=1e-5)


def test_cross_level_diagnostics(dims: tuple[int, int, int]) -> None:
    d_k, d_v, num_levels = dims
    state = init_associative_memory(d_k, d_v, AssociativeMemoryConfig(num_levels=num_levels))
    key = np.asarray(normalize_key(jnp.ones(d_k)), dtype=np.float32)
    levels = query_memory_levels(key, state)
    cross = compute_cross_level_diagnostics(levels)
    assert cross.pairwise_cosine.shape == (num_levels, num_levels)
    assert cross.response_variance.shape == (d_v,)
    assert cross.magnitude_variance >= 0


def test_historical_checkpoints(memory_bundle) -> None:
    state, bundle = memory_bundle
    key = np.asarray(normalize_key(jnp.asarray(np.random.default_rng(4).standard_normal(16))), dtype=np.float32)
    history = query_historical_checkpoints(key, bundle)
    assert len(history) == bundle.T
    snap = history[0]
    for j, z in enumerate(snap.level_responses):
        manual = np.asarray(bundle.state_at(0).matrices[j] @ jnp.asarray(key), dtype=np.float32)
        np.testing.assert_allclose(z, manual, rtol=1e-5, atol=1e-6)


def test_deploy_sample_no_labels(resnet_params_and_image, resnet_memory_bundle) -> None:
    params, x = resnet_params_and_image
    state, bundle = resnet_memory_bundle
    rec = deploy_sample(
        params,
        x,
        state,
        sample_id="test-0",
        num_classes=7,
        checkpoints=bundle,
        include_history=True,
    )
    assert rec.sample_id == "test-0"
    assert rec.prediction.probabilities.shape == (7,)
    assert rec.representation.key.shape[0] == rec.representation.h.shape[0]
    assert len(rec.memory_levels) == len(state.matrices)
    assert rec.optional_history is not None
    d = rec.to_dict()
    assert "memory" in d
    assert "optional_history" in d
    assert "error" not in d


def test_batch_matches_single(resnet_params_and_image, resnet_memory_bundle) -> None:
    params, x = resnet_params_and_image
    state, bundle = resnet_memory_bundle
    rng = np.random.default_rng(5)
    batch = np.stack([x, rng.random((28, 28, 3), dtype=np.float32)], axis=0)
    batch_recs = deploy_batch(
        params,
        batch,
        state,
        sample_ids=[0, 1],
        num_classes=7,
        checkpoints=bundle,
        include_history=False,
    )
    single = deploy_sample(
        params,
        batch[0],
        state,
        sample_id=0,
        num_classes=7,
        include_history=False,
    )
    np.testing.assert_allclose(
        batch_recs[0].memory_levels[0].response,
        single.memory_levels[0].response,
        rtol=1e-5,
        atol=1e-6,
    )


def test_sanity_checks(resnet_params_and_image, resnet_memory_bundle) -> None:
    params, x = resnet_params_and_image
    state, bundle = resnet_memory_bundle
    batch = np.stack([x, x], axis=0)
    report = run_deployment_sanity_checks(
        params,
        batch,
        state,
        num_classes=7,
        checkpoints=bundle,
    )
    assert report.checks["key_unit_norm"]
    assert report.checks["batch_matches_single"]
    assert report.checks["manual_matmul_matches"]
    assert report.passed


def test_representation_key_norm(resnet_params_and_image, resnet_memory_bundle) -> None:
    params, x = resnet_params_and_image
    state, _ = resnet_memory_bundle
    rec = deploy_sample(params, x, state, sample_id=0, num_classes=7)
    assert abs(float(np.linalg.norm(rec.representation.key)) - 1.0) < 1e-3


def test_records_to_dataframe_and_save(tmp_path: Path, resnet_params_and_image, resnet_memory_bundle) -> None:
    params, x = resnet_params_and_image
    state, bundle = resnet_memory_bundle
    rec = deploy_sample(
        params,
        x,
        state,
        sample_id=0,
        num_classes=7,
        checkpoints=bundle,
        include_history=True,
    )
    df = records_to_dataframe([rec])
    assert "z1_magnitude" in df.columns
    assert "normalized_entropy" in df.columns
    paths = save_deployment_records([rec], tmp_path)
    assert paths["jsonl"].exists()
    assert paths["csv"].exists()
    line = paths["jsonl"].read_text(encoding="utf-8").strip()
    payload = json.loads(line)
    assert payload["sample_id"] == 0


def test_offline_evaluation_smoke(resnet_params_and_image, resnet_memory_bundle) -> None:
    params, x = resnet_params_and_image
    state, bundle = resnet_memory_bundle
    rng = np.random.default_rng(6)
    n = 40
    batch = np.stack([rng.random((28, 28, 3), dtype=np.float32) for _ in range(n)], axis=0)
    labels = rng.integers(0, 7, size=n)
    recs = deploy_batch(
        params,
        batch,
        state,
        sample_ids=list(range(n)),
        num_classes=7,
        include_history=False,
    )
    df = records_to_dataframe(recs)
    df = join_ground_truth(df, labels=labels)
    cal = df.iloc[:20].copy()
    te = df.iloc[20:].copy()
    result = offline_evaluate_memory_information(cal, te)
    assert len(result) >= 3
    assert "prediction_only_H" in result["model"].values
    mem_cols = memory_diagnostic_feature_columns(df)
    assert mem_cols


def test_deploy_does_not_use_value_from_logits(resnet_params_and_image, resnet_memory_bundle) -> None:
    """Deployment record must not contain training value v = y - p."""
    params, x = resnet_params_and_image
    state, _ = resnet_memory_bundle
    rec = deploy_sample(params, x, state, sample_id=0, num_classes=7)
    blob = json.dumps(rec.to_dict())
    assert "value_from_logits" not in blob
    assert "y_minus_p" not in blob
