"""Sanity checks for correctness memory (d_v x d_k matrices)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from optimizer.associative_memory import AssociativeMemoryConfig, normalize_key
from optimizer.correctness_memory import (
    CorrectnessMemory,
    batch_correctness_delta_rule_step,
    correctness_delta_rule_step,
    correctness_from_logits,
    correctness_retrieve,
    correctness_value_projection_matrix,
    correctness_value_vector,
    expected_matrix_shape,
    init_correctness_memory,
)
from research.common.correctness_clinical_training import per_sample_correctness_batch
from research.common.resnet import init_resnet18_params, resnet18_apply, resnet18_features


@pytest.fixture
def dims() -> tuple[int, int, int]:
    return 16, 7, 4  # d_k, d_v, num_levels


def test_matrix_and_key_shapes(dims: tuple[int, int, int]) -> None:
    d_k, d_v, num_levels = dims
    cfg = AssociativeMemoryConfig(num_levels=num_levels, correctness_z_dim=d_v)
    state = init_correctness_memory(d_k, cfg)
    for matrix in state.matrices:
        assert matrix.shape == expected_matrix_shape(d_k, d_v)
    k = normalize_key(jnp.ones((d_k,), dtype=jnp.float32))
    z = correctness_retrieve(k, state.matrices[0])
    assert np.asarray(z).shape == (d_v,)


def test_correctness_label_binary() -> None:
    logits = jnp.array([2.0, -1.0, 0.5], dtype=jnp.float32)
    labels = jnp.array(0, dtype=jnp.int32)
    c = correctness_from_logits(logits, labels)
    assert float(c) in (0.0, 1.0)


def test_value_vector_matches_spec(dims: tuple[int, int, int]) -> None:
    d_k, d_v, _ = dims
    k = normalize_key(jnp.arange(d_k, dtype=jnp.float32))
    W = correctness_value_projection_matrix(d_k, d_v, seed=0)
    u1 = correctness_value_vector(k, 1.0, W)
    u0 = correctness_value_vector(k, 0.0, W)
    assert np.allclose(np.asarray(u1), np.asarray(W @ k))
    assert np.allclose(np.asarray(u0), 0.0)


def test_update_direction_matches_squared_error_gradient(dims: tuple[int, int, int]) -> None:
    d_k, d_v, _ = dims
    rng = np.random.default_rng(0)
    matrix = jnp.zeros((d_v, d_k), dtype=jnp.float32)
    k = normalize_key(jnp.asarray(rng.standard_normal(d_k), dtype=jnp.float32))
    c = 1.0
    eta = 0.1
    W = correctness_value_projection_matrix(d_k, d_v, seed=0)
    updated = correctness_delta_rule_step(matrix, k, c, eta, W)
    u = np.asarray(correctness_value_vector(k, c, W))
    pred = np.asarray(matrix @ k.reshape(-1))
    expected = matrix + eta * np.outer(u - pred, np.asarray(k.reshape(-1)))
    assert np.allclose(np.asarray(updated), np.asarray(expected), rtol=1e-5, atol=1e-6)


def test_batch_update_runs(dims: tuple[int, int, int]) -> None:
    d_k, d_v, _ = dims
    rng = np.random.default_rng(1)
    matrix = jnp.zeros((d_v, d_k), dtype=jnp.float32)
    k_batch = jnp.asarray(rng.standard_normal((8, d_k)), dtype=jnp.float32)
    k_batch = normalize_key(k_batch)
    logits = jnp.asarray(rng.standard_normal((8, 3)), dtype=jnp.float32)
    labels = jnp.asarray(rng.integers(0, 3, size=8), dtype=jnp.int32)
    c_batch = correctness_from_logits(logits, labels)
    W = correctness_value_projection_matrix(d_k, d_v, seed=0)
    updated = batch_correctness_delta_rule_step(matrix, k_batch, c_batch, eta=0.05, projection=W)
    assert updated.shape == (d_v, d_k)
    assert np.all(np.isfinite(np.asarray(updated)))


def test_trained_memory_finite(dims: tuple[int, int, int]) -> None:
    d_k, d_v, num_levels = dims
    mem = CorrectnessMemory(d_k, AssociativeMemoryConfig(num_levels=num_levels, correctness_z_dim=d_v))
    rng = np.random.default_rng(2)
    for _ in range(20):
        logits = jnp.asarray(rng.standard_normal(5), dtype=jnp.float32)
        label = int(rng.integers(0, 5))
        c = correctness_from_logits(logits, label)
        k = normalize_key(jnp.asarray(rng.standard_normal(d_k), dtype=jnp.float32))
        mem.update(k, c)
    for matrix in mem.state.matrices:
        assert matrix.shape == expected_matrix_shape(d_k, d_v)
        assert np.all(np.isfinite(np.asarray(matrix)))


def test_per_sample_correctness_batch_matches_single_forwards() -> None:
    key = jax.random.PRNGKey(0)
    params = init_resnet18_params(key, num_classes=7)
    rng = np.random.default_rng(3)
    x_batch = jnp.asarray(rng.standard_normal((4, 28, 28, 3)), dtype=jnp.float32)
    y_batch = jnp.asarray(rng.integers(0, 7, size=4), dtype=jnp.int32)

    k_batch, c_batch = per_sample_correctness_batch(params, x_batch, y_batch, eps=1e-8)

    for i in range(4):
        logits_i = resnet18_apply(params, x_batch[i])
        h_i = resnet18_features(params, x_batch[i])
        k_i = normalize_key(h_i, eps=1e-8)
        c_i = correctness_from_logits(logits_i, y_batch[i])
        assert np.allclose(np.asarray(k_batch[i]), np.asarray(k_i), rtol=1e-5, atol=1e-5)
        assert float(c_batch[i]) == float(c_i)

    logits_b = resnet18_apply(params, x_batch)
    c_batched = correctness_from_logits(logits_b, y_batch)
    assert not np.allclose(np.asarray(c_batch), np.asarray(c_batched))
