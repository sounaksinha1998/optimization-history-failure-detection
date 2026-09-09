"""Numerical sanity checks for associative memory (run before full DermaMNIST)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from optimizer.associative_memory import (
    AssociativeMemory,
    AssociativeMemoryConfig,
    AssociativeMemoryNaNError,
    AssociativeMemoryState,
    associative_retrieve,
    associative_update,
    batch_delta_rule_step,
    delta_rule_step,
    init_associative_memory,
    normalize_key,
    randomize_associative_memory,
    retrieve_all_levels,
    value_from_logits,
)
from research.common.associative_attention import retrieve_z_memory, retrieve_z_memory_from_trajectory
from research.common.memory import (
    load_associative_artifacts,
    memory_reconstruction_diagnostics,
    save_frozen_associative_artifacts,
)
from research.common.msa import batch_associative_z_memory, per_sample_associative_signals
from research.common.spectral_memory import temporal_fft_features
from research.common.trajectory_store import (
    MemoryCheckpointStore,
    MissingMemoryCheckpointsError,
    build_query_trajectory,
)
from research.memory_identification.memory_identification import _z_memory_from_randomized_trajectories


@pytest.fixture
def dims() -> tuple[int, int, int]:
    return 8, 5, 4  # d_k, d_v, num_levels


def _trained_state(d_k: int, d_v: int, num_levels: int, steps: int, seed: int) -> AssociativeMemory:
    mem = AssociativeMemory(d_k, d_v, AssociativeMemoryConfig(num_levels=num_levels))
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        mem.update(
            normalize_key(jnp.asarray(rng.standard_normal(d_k), dtype=jnp.float32)),
            jnp.asarray(rng.standard_normal(d_v), dtype=jnp.float32),
        )
    return mem


def _bundle_from_memories(
    states: list[AssociativeMemoryState],
    steps: list[int],
    *,
    sample_every: int = 1,
):
    store = MemoryCheckpointStore(sample_every=sample_every)
    for step, state in zip(steps, states, strict=True):
        if store.should_record(step):
            store.save_memory_checkpoint(step, state)
        else:
            store.save_final_checkpoint(step, state)
    return store.freeze()


def test_key_value_dims(dims: tuple[int, int, int]) -> None:
    d_k, d_v, num_levels = dims
    cfg = AssociativeMemoryConfig(num_levels=num_levels)
    state = init_associative_memory(d_k, d_v, cfg)
    k = normalize_key(jnp.ones((d_k,), dtype=jnp.float32) * 0.5)
    v = jnp.ones((d_v,), dtype=jnp.float32) * 0.2
    assert k.shape == (d_k,)
    assert v.shape == (d_v,)
    for matrix in state.matrices:
        assert matrix.shape == (d_v, d_k)


def test_single_example_delta_rule(dims: tuple[int, int, int]) -> None:
    d_k, d_v, _ = dims
    rng = np.random.default_rng(0)
    matrix = jnp.asarray(rng.standard_normal((d_v, d_k)), dtype=jnp.float32)
    k = normalize_key(jnp.asarray(rng.standard_normal(d_k), dtype=jnp.float32))
    v = jnp.asarray(rng.standard_normal(d_v), dtype=jnp.float32)
    eta = 0.1
    got = np.asarray(delta_rule_step(matrix, k, v, eta))
    err = np.asarray(matrix @ k - v)
    expected = np.asarray(matrix) - eta * np.outer(err, np.asarray(k))
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_batch_averaging_not_sum(dims: tuple[int, int, int]) -> None:
    d_k, d_v, _ = dims
    rng = np.random.default_rng(1)
    matrix = jnp.asarray(rng.standard_normal((d_v, d_k)), dtype=jnp.float32)
    k_batch = normalize_key(jnp.asarray(rng.standard_normal((4, d_k)), dtype=jnp.float32))
    v_batch = jnp.asarray(rng.standard_normal((4, d_v)), dtype=jnp.float32)
    eta = 0.05
    batch_updated = np.asarray(batch_delta_rule_step(matrix, k_batch, v_batch, eta))
    grads = []
    for i in range(4):
        err = np.asarray(matrix @ k_batch[i] - v_batch[i])
        grads.append(np.outer(err, np.asarray(k_batch[i])))
    mean_grad = np.mean(np.stack(grads, axis=0), axis=0)
    expected = np.asarray(matrix) - eta * mean_grad
    np.testing.assert_allclose(batch_updated, expected, rtol=1e-5, atol=1e-6)
    summed = np.asarray(matrix) - eta * np.sum(np.stack(grads, axis=0), axis=0)
    assert not np.allclose(batch_updated, summed, rtol=1e-3)


def test_batch_size_one_matches_single_example(dims: tuple[int, int, int]) -> None:
    d_k, d_v, _ = dims
    rng = np.random.default_rng(2)
    matrix = jnp.asarray(rng.standard_normal((d_v, d_k)), dtype=jnp.float32)
    k = normalize_key(jnp.asarray(rng.standard_normal(d_k), dtype=jnp.float32))
    v = jnp.asarray(rng.standard_normal(d_v), dtype=jnp.float32)
    eta = 0.07
    single = np.asarray(delta_rule_step(matrix, k, v, eta))
    batched = np.asarray(batch_delta_rule_step(matrix, k[None, :], v[None, :], eta))
    np.testing.assert_allclose(single, batched, rtol=1e-5, atol=1e-6)


def test_matrix_shape_preserved_after_update(dims: tuple[int, int, int]) -> None:
    d_k, d_v, num_levels = dims
    cfg = AssociativeMemoryConfig(num_levels=num_levels)
    mem = AssociativeMemory(d_k, d_v, cfg)
    k = normalize_key(jnp.arange(d_k, dtype=jnp.float32))
    v = jnp.arange(d_v, dtype=jnp.float32) * 0.1
    mem.update(k, v)
    for matrix in mem.state.matrices:
        assert matrix.shape == (d_v, d_k)


def test_outputs_finite(dims: tuple[int, int, int]) -> None:
    d_k, d_v, num_levels = dims
    rng = np.random.default_rng(0)
    mem = AssociativeMemory(d_k, d_v, AssociativeMemoryConfig(num_levels=num_levels))
    for _ in range(20):
        k = normalize_key(jnp.asarray(rng.standard_normal(d_k), dtype=jnp.float32))
        v = jnp.asarray(rng.standard_normal(d_v), dtype=jnp.float32)
        mem.update(k, v)
    out = mem.retrieve(normalize_key(jnp.asarray(rng.standard_normal(d_k), dtype=jnp.float32)))
    assert jnp.all(jnp.isfinite(out))


def test_nan_update_fails_loudly(dims: tuple[int, int, int]) -> None:
    d_k, d_v, num_levels = dims
    mem = AssociativeMemory(d_k, d_v, AssociativeMemoryConfig(num_levels=num_levels))
    k = normalize_key(jnp.ones((d_k,), dtype=jnp.float32))
    v = jnp.ones((d_v,), dtype=jnp.float32)
    v = v.at[0].set(jnp.nan)
    with pytest.raises(AssociativeMemoryNaNError, match="Non-finite"):
        mem.update(k, v)


def test_levels_differ_after_training(dims: tuple[int, int, int]) -> None:
    d_k, d_v, _ = dims
    mem = AssociativeMemory(d_k, d_v, AssociativeMemoryConfig(num_levels=4))
    rng = np.random.default_rng(1)
    for _ in range(128):
        k = normalize_key(jnp.asarray(rng.standard_normal(d_k), dtype=jnp.float32))
        v = jnp.asarray(rng.standard_normal(d_v), dtype=jnp.float32)
        mem.update(k, v)
    k = normalize_key(jnp.asarray(rng.standard_normal(d_k), dtype=jnp.float32))
    levels = [np.asarray(mem.retrieve_level(k, j)) for j in range(4)]
    assert not np.allclose(levels[0], levels[1])
    assert not np.allclose(levels[2], levels[3])


def test_eta_zero_memory_unchanged(dims: tuple[int, int, int]) -> None:
    d_k, d_v, num_levels = dims
    cfg = AssociativeMemoryConfig(
        num_levels=num_levels,
        learning_rates=tuple(0.0 for _ in range(num_levels)),
    )
    state = init_associative_memory(d_k, d_v, cfg)
    k = normalize_key(jnp.arange(d_k, dtype=jnp.float32))
    v = jnp.arange(d_v, dtype=jnp.float32)
    new_state = associative_update(state, k, v, cfg)
    for before, after in zip(state.matrices, new_state.matrices):
        assert jnp.allclose(before, after)


def test_single_timestep_fft_trivial() -> None:
    R = np.random.default_rng(2).standard_normal((1, 4, 5)).astype(np.float32)
    S = temporal_fft_features(R, use_fft=True)
    assert S.shape == (4 * 1, 5)
    assert np.all(np.isfinite(S))


def test_fft_axis_is_training_time() -> None:
    t_steps, k_levels, d_v = 8, 2, 3
    R = np.zeros((t_steps, k_levels, d_v), dtype=np.float32)
    t = np.arange(t_steps, dtype=np.float32)
    R[:, 0, 0] = np.sin(2.0 * np.pi * t / t_steps)
    spectrum = np.abs(np.fft.rfft(R, axis=0))
    S = temporal_fft_features(R, use_fft=True)
    t_freq = t_steps // 2 + 1
    assert spectrum.shape[0] == t_freq
    np.testing.assert_allclose(spectrum[:, 0, 0], np.abs(np.fft.rfft(R[:, 0, 0])), rtol=1e-5, atol=1e-6)
    assert S.shape == (k_levels * t_freq, d_v)
    with pytest.raises(ValueError, match="share checkpoint length"):
        temporal_fft_features(R[None, ...], use_fft=True, lengths=np.array([3]))


def test_fixed_seed_deterministic_retrieval(dims: tuple[int, int, int]) -> None:
    d_k, d_v, _ = dims
    h = np.random.default_rng(3).standard_normal(d_k).astype(np.float32)
    R = np.random.default_rng(4).standard_normal((6, 4, d_v)).astype(np.float32)
    cfg = AssociativeMemoryConfig(use_fft=True, use_attention=False)
    z1, _ = retrieve_z_memory_from_trajectory(h, R, cfg=cfg, seed=99, d_out=16)
    z2, _ = retrieve_z_memory_from_trajectory(h, R, cfg=cfg, seed=99, d_out=16)
    assert np.allclose(z1, z2)


def test_deploy_path_no_label_access(dims: tuple[int, int, int]) -> None:
    d_k, d_v, _ = dims
    h = np.random.default_rng(5).standard_normal(d_k).astype(np.float32)
    R = np.random.default_rng(6).standard_normal((8, 4, d_v)).astype(np.float32)
    signals = per_sample_associative_signals(h, R, AssociativeMemoryConfig(use_attention=False), d_out=12)
    assert "z_0" in signals
    assert np.isfinite(signals["z_memory_norm"])


def test_attention_without_trained_params_raises(dims: tuple[int, int, int]) -> None:
    d_k, d_v, _ = dims
    h = np.ones((d_k,), dtype=np.float32)
    S = np.ones((4, d_v), dtype=np.float32)
    with pytest.raises(ValueError, match="attention_params"):
        retrieve_z_memory(h, S, use_attention=True, d_out=8)


def test_value_from_logits_shape() -> None:
    logits = jnp.array([1.0, 0.0, -1.0], dtype=jnp.float32)
    labels = jnp.array(0, dtype=jnp.int32)
    v = value_from_logits(logits, labels, num_classes=3)
    assert v.shape == (3,)
    assert np.isclose(float(v.sum()), 0.0, atol=1e-5)


def test_frozen_query_trajectory_ignores_historical_keys(dims: tuple[int, int, int]) -> None:
    d_k, d_v, num_levels = dims
    cfg = AssociativeMemoryConfig(
        num_levels=num_levels,
        update_every=tuple(1 for _ in range(num_levels)),
        learning_rates=tuple(0.1 for _ in range(num_levels)),
    )
    mem1 = AssociativeMemory(d_k, d_v, cfg)
    mem2 = AssociativeMemory(d_k, d_v, cfg)
    rng = np.random.default_rng(1)
    for mem, offset in ((mem1, 0), (mem2, 99)):
        local = np.random.default_rng(offset)
        for _ in range(8):
            mem.update(
                normalize_key(jnp.asarray(local.standard_normal(d_k), dtype=jnp.float32)),
                jnp.asarray(local.standard_normal(d_v), dtype=jnp.float32),
            )
    bundle = _bundle_from_memories([mem1.state, mem2.state], [1, 2])
    rng = np.random.default_rng(20)
    h_T = rng.standard_normal(d_k).astype(np.float32)
    k_T = np.asarray(normalize_key(jnp.asarray(h_T)))
    k_hist = np.asarray(normalize_key(jnp.asarray(rng.standard_normal(d_k), dtype=jnp.float32)))
    R = build_query_trajectory(h_T, bundle)
    assert R.shape == (2, num_levels, d_v)
    for level in range(num_levels):
        np.testing.assert_allclose(
            R[0, level],
            np.asarray(associative_retrieve(jnp.asarray(k_T), mem1.state.matrices[level])),
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            R[1, level],
            np.asarray(associative_retrieve(jnp.asarray(k_T), mem2.state.matrices[level])),
            rtol=1e-5,
            atol=1e-6,
        )
        hist_row = np.asarray(associative_retrieve(jnp.asarray(k_hist), mem1.state.matrices[level]))
        assert not np.allclose(R[0, level], hist_row)


def test_shared_checkpoint_grid(dims: tuple[int, int, int]) -> None:
    d_k, d_v, num_levels = dims
    mem_a = _trained_state(d_k, d_v, num_levels, steps=4, seed=3)
    mem_b = _trained_state(d_k, d_v, num_levels, steps=6, seed=4)
    mem_c = _trained_state(d_k, d_v, num_levels, steps=8, seed=5)
    bundle = _bundle_from_memories([mem_a.state, mem_b.state, mem_c.state], [2, 4, 6])
    h = np.random.default_rng(21).standard_normal((5, d_k)).astype(np.float32)
    R = build_query_trajectory(h, bundle)
    assert R.shape == (5, 3, num_levels, d_v)
    assert np.array_equal(bundle.checkpoint_steps, np.array([2, 4, 6]))
    for i in range(5):
        assert R[i].shape[0] == bundle.T


def test_no_t1_fallback(dims: tuple[int, int, int]) -> None:
    d_k, d_v, num_levels = dims
    mem_a = _trained_state(d_k, d_v, num_levels, steps=4, seed=6)
    mem_b = _trained_state(d_k, d_v, num_levels, steps=8, seed=7)
    bundle = _bundle_from_memories([mem_a.state, mem_b.state], [1, 2])
    h = np.random.default_rng(22).standard_normal((3, d_k)).astype(np.float32)
    z, _ = batch_associative_z_memory(h, bundle, AssociativeMemoryConfig(use_attention=False), d_out=8)
    R = build_query_trajectory(h, bundle)
    assert R.shape[1] > 1
    assert z.shape == (3, 8)
    empty = MemoryCheckpointStore(sample_every=1)
    with pytest.raises(MissingMemoryCheckpointsError, match="No associative memory checkpoints"):
        empty.freeze()


def test_z_memory_independent_of_sample_id(dims: tuple[int, int, int]) -> None:
    d_k, d_v, num_levels = dims
    mem_a = _trained_state(d_k, d_v, num_levels, steps=4, seed=8)
    mem_b = _trained_state(d_k, d_v, num_levels, steps=8, seed=9)
    bundle = _bundle_from_memories([mem_a.state, mem_b.state], [1, 2])
    h = np.random.default_rng(23).standard_normal((2, d_k)).astype(np.float32)
    cfg = AssociativeMemoryConfig(use_attention=False)
    z1, _ = batch_associative_z_memory(
        h, bundle, cfg, sample_ids=np.array(["a", "b"], dtype=object), d_out=8
    )
    z2, _ = batch_associative_z_memory(
        h, bundle, cfg, sample_ids=np.array(["zzzz", "yyyy"], dtype=object), d_out=8
    )
    np.testing.assert_allclose(z1, z2)


def test_save_load_roundtrip(dims: tuple[int, int, int]) -> None:
    d_k, d_v, num_levels = dims
    mem = _trained_state(d_k, d_v, num_levels, steps=10, seed=8)
    store = MemoryCheckpointStore(sample_every=1)
    store.save_memory_checkpoint(1, mem.state)
    bundle = store.freeze()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        save_frozen_associative_artifacts(
            root,
            task="dermamnist",
            seed=42,
            state=mem.state,
            checkpoints=bundle,
            cfg=AssociativeMemoryConfig(num_levels=num_levels),
        )
        loaded_state, loaded_bundle, loaded_cfg = load_associative_artifacts(root, "dermamnist", 42)
        for a, b in zip(mem.state.matrices, loaded_state.matrices):
            assert jnp.allclose(a, b)
        assert loaded_bundle.T == bundle.T
        assert loaded_cfg.num_levels == num_levels


def test_memory_reconstruction_diagnostics(dims: tuple[int, int, int]) -> None:
    d_k, d_v, num_levels = dims
    mem = AssociativeMemory(d_k, d_v, AssociativeMemoryConfig(num_levels=num_levels))
    rng = np.random.default_rng(9)
    keys = []
    values = []
    for _ in range(16):
        k = normalize_key(jnp.asarray(rng.standard_normal(d_k), dtype=jnp.float32))
        v = jnp.asarray(rng.standard_normal(d_v), dtype=jnp.float32)
        mem.update(k, v)
        keys.append(np.asarray(k))
        values.append(np.asarray(v))
    diag = memory_reconstruction_diagnostics(mem.state, np.stack(keys), np.stack(values))
    assert {"mse", "cosine", "level"}.issubset(diag.columns)
    assert len(diag) == 16 * num_levels


def test_randomize_associative_memory_norm_matched(dims: tuple[int, int, int]) -> None:
    d_k, d_v, num_levels = dims
    mem = _trained_state(d_k, d_v, num_levels, steps=64, seed=10)
    rand_state = randomize_associative_memory(mem.state, seed=123)
    for actual, rnd in zip(mem.state.matrices, rand_state.matrices):
        actual_arr = np.asarray(actual)
        rnd_arr = np.asarray(rnd)
        assert np.isclose(np.linalg.norm(actual_arr), np.linalg.norm(rnd_arr), rtol=1e-5)
        if np.linalg.norm(actual_arr) > 1e-4:
            assert not jnp.allclose(actual, rnd)


def test_batch_associative_z_memory_label_free(dims: tuple[int, int, int]) -> None:
    d_k, d_v, num_levels = dims
    mem_a = _trained_state(d_k, d_v, num_levels, steps=4, seed=11)
    mem_b = _trained_state(d_k, d_v, num_levels, steps=8, seed=12)
    bundle = _bundle_from_memories([mem_a.state, mem_b.state], [1, 2])
    h = np.random.default_rng(11).standard_normal((3, d_k)).astype(np.float32)
    z, attn = batch_associative_z_memory(h, bundle, AssociativeMemoryConfig(use_attention=False), d_out=8)
    assert z.shape == (3, 8)
    assert attn.shape[0] == 3
    assert np.all(np.isfinite(z))


def test_z_memory_from_randomized_checkpoints(dims: tuple[int, int, int]) -> None:
    d_k, d_v, num_levels = dims
    mem_a = _trained_state(d_k, d_v, num_levels, steps=4, seed=14)
    mem_b = _trained_state(d_k, d_v, num_levels, steps=8, seed=15)
    bundle = _bundle_from_memories([mem_a.state, mem_b.state], [1, 2])
    h = np.random.default_rng(14).standard_normal((2, d_k)).astype(np.float32)
    z = _z_memory_from_randomized_trajectories(
        h,
        bundle,
        seed=99,
        assoc_cfg=AssociativeMemoryConfig(use_attention=False),
        d_out=8,
    )
    assert z.shape == (2, 8)
    assert np.all(np.isfinite(z))


def test_associative_retrieve_batched(dims: tuple[int, int, int]) -> None:
    d_k, d_v, num_levels = dims
    state = init_associative_memory(d_k, d_v, AssociativeMemoryConfig(num_levels=num_levels))
    matrix = state.matrices[0]
    k_batch = normalize_key(jnp.asarray(np.random.default_rng(12).standard_normal((3, d_k)), dtype=jnp.float32))
    out = associative_retrieve(k_batch, matrix)
    assert out.shape == (3, d_v)
    _ = retrieve_all_levels
