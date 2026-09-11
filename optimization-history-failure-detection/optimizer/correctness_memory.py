"""High-dimensional correctness associative memory.

Implementation equations (audit):

    k_t = h_t / (||h_t||_2 + eps)                         key, R^{d_k}
    c_t = 1[y_hat_t = y_t]                                 target in {0, 1}
    u_t = c_t P k_t                                        value, R^{d_v}
    M^{(j)} in R^{d_v x d_k}
    z_t^{(j)} = M^{(j)} k_t                                prediction u-hat, R^{d_v}
    L_M^{(j)} = (1/2) ||M^{(j)} k_t - c_t P k_t||_2^2
    grad_M = (M^{(j)} k_t - c_t P k_t) k_t^T
    M_{t+1}^{(j)} = M_t^{(j)} + eta_j (c_t P k_t - M_t^{(j)} k_t) k_t^T

When d_v = d_k, P = I so u_t = c_t k_t (full key-direction value).

Deploy (frozen h_T):

    k_T(x) = h_T(x) / (||h_T(x)||_2 + eps)
    z_j(x) = M_T^{(j)} k_T(x) in R^{d_v}
    z_combined(x) = [z_1, ..., z_4] in R^{4 d_v}

Probe: cal-fit logistic on z_combined predicts error; s_fail = P(error | z_combined).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from optimizer.associative_memory import (
    AssociativeMemoryConfig,
    assert_finite_array,
    batch_delta_rule_step,
    delta_rule_step,
    normalize_key,
    resolve_level_schedule,
)

NUM_LEVELS_DEFAULT = 4


class CorrectnessMemoryState(NamedTuple):
    """Frozen-friendly state: one (d_v, d_k) matrix per level."""

    matrices: tuple[jnp.ndarray, ...]
    step: jnp.ndarray


def expected_matrix_shape(d_k: int, d_v: int) -> tuple[int, int]:
    return (int(d_v), int(d_k))


def resolve_correctness_dims(
    d_k: int,
    cfg: AssociativeMemoryConfig | None = None,
) -> tuple[int, int]:
    cfg = cfg or AssociativeMemoryConfig()
    return int(d_k), int(cfg.correctness_z_dim)


def correctness_value_projection_matrix(
    d_k: int,
    d_v: int,
    *,
    seed: int = 0,
) -> jnp.ndarray:
    """Fixed map P in R^{d_v x d_k} for u_t = c_t P k_t. Identity when d_v == d_k."""
    if int(d_v) == int(d_k):
        return jnp.eye(int(d_k), dtype=jnp.float32)
    rng = np.random.default_rng(int(seed))
    W = rng.standard_normal((int(d_v), int(d_k))).astype(np.float32)
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    return jnp.asarray(W / (norms + 1e-8), dtype=jnp.float32)


def init_correctness_memory(
    d_k: int,
    cfg: AssociativeMemoryConfig | None = None,
) -> CorrectnessMemoryState:
    """Initialize all M^j to zeros with shape (d_v, d_k)."""
    cfg = cfg or AssociativeMemoryConfig()
    _, d_v = resolve_correctness_dims(d_k, cfg)
    shape = expected_matrix_shape(d_k, d_v)
    matrices = tuple(jnp.zeros(shape, dtype=jnp.float32) for _ in range(cfg.num_levels))
    return CorrectnessMemoryState(matrices=matrices, step=jnp.array(0, dtype=jnp.int32))


def correctness_from_logits(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """c_t = 1[argmax(logits) == y_t], shape () or (batch,)."""
    logits = jnp.asarray(logits, dtype=jnp.float32)
    labels = jnp.asarray(labels, dtype=jnp.int32)
    if logits.ndim == 1:
        pred = int(jnp.argmax(logits))
        return jnp.asarray(1.0 if pred == int(labels) else 0.0, dtype=jnp.float32)
    preds = jnp.argmax(logits, axis=-1)
    return (preds == labels).astype(jnp.float32)


def correctness_value_vector(
    k: jnp.ndarray,
    c: jnp.ndarray | float,
    projection: jnp.ndarray,
) -> jnp.ndarray:
    """u_t = c_t P k_t."""
    k_arr = jnp.asarray(k, dtype=jnp.float32).reshape(-1)
    c_arr = jnp.asarray(c, dtype=jnp.float32).reshape(())
    proj = jnp.asarray(projection, dtype=jnp.float32)
    return c_arr * (proj @ k_arr)


def correctness_retrieve(k: jnp.ndarray, matrix: jnp.ndarray) -> jnp.ndarray:
    """z = M k; returns shape (d_v,) or (batch, d_v)."""
    k = jnp.asarray(k, dtype=jnp.float32)
    matrix = jnp.asarray(matrix, dtype=jnp.float32)
    if k.ndim == 1:
        return matrix @ k
    return k @ matrix.T


def retrieve_all_correctness_levels(
    k: jnp.ndarray,
    state: CorrectnessMemoryState,
) -> jnp.ndarray:
    """z^j(k) for all levels; shape (num_levels, d_v) or (batch, num_levels, d_v)."""
    outputs = [correctness_retrieve(k, m) for m in state.matrices]
    stacked = jnp.stack(outputs, axis=0 if k.ndim == 1 else 1)
    return assert_finite_array(stacked, name="retrieve_all_correctness_levels")


def correctness_delta_rule_step(
    matrix: jnp.ndarray,
    k: jnp.ndarray,
    c: jnp.ndarray | float,
    eta: float,
    projection: jnp.ndarray,
) -> jnp.ndarray:
    """M' = M - eta (M k - c P k) k^T."""
    u = correctness_value_vector(k, c, projection)
    return delta_rule_step(matrix, k, u, eta)


def batch_correctness_delta_rule_step(
    matrix: jnp.ndarray,
    k_batch: jnp.ndarray,
    c_batch: jnp.ndarray,
    eta: float,
    projection: jnp.ndarray,
) -> jnp.ndarray:
    """Mean-aggregated delta on u = c P k."""
    k_batch = jnp.asarray(k_batch, dtype=jnp.float32)
    c_batch = jnp.asarray(c_batch, dtype=jnp.float32).reshape(-1)
    proj = jnp.asarray(projection, dtype=jnp.float32)
    if k_batch.ndim != 2 or c_batch.ndim != 1:
        raise ValueError("k_batch must be (batch, d_k) and c_batch (batch,)")
    v_batch = c_batch[:, None] * (k_batch @ proj.T)
    return batch_delta_rule_step(matrix, k_batch, v_batch, eta)


def correctness_update(
    state: CorrectnessMemoryState,
    k: jnp.ndarray,
    c: jnp.ndarray,
    cfg: AssociativeMemoryConfig | None = None,
    *,
    d_k: int | None = None,
) -> CorrectnessMemoryState:
    """Advance correctness memory by one global step using target c_t in {0, 1}."""
    cfg = cfg or AssociativeMemoryConfig()
    if d_k is None:
        d_k = int(state.matrices[0].shape[1])
    _, d_v = resolve_correctness_dims(d_k, cfg)
    projection = correctness_value_projection_matrix(
        d_k,
        d_v,
        seed=cfg.aggregation_seed,
    )
    update_every, learning_rates = resolve_level_schedule(cfg)
    step = int(state.step) + 1
    k_arr = jnp.asarray(k, dtype=jnp.float32)
    c_arr = jnp.asarray(c, dtype=jnp.float32)
    batch_size = 1 if k_arr.ndim == 1 else int(k_arr.shape[0])
    new_matrices: list[jnp.ndarray] = []

    for level_idx, (matrix, every, eta) in enumerate(
        zip(state.matrices, update_every, learning_rates),
        start=1,
    ):
        if step % every != 0 or eta == 0.0:
            new_matrices.append(matrix)
            continue
        before = np.asarray(matrix, dtype=np.float32)
        if k_arr.ndim == 1:
            updated = correctness_delta_rule_step(matrix, k_arr, c_arr, eta, projection)
            k_vec = np.asarray(k_arr.reshape(-1))
            u = np.asarray(correctness_value_vector(k_arr, c_arr, projection))
            err = np.asarray(before @ k_vec - u)
            grad_norm = float(np.linalg.norm(np.outer(err, k_vec)))
        else:
            updated = batch_correctness_delta_rule_step(matrix, k_arr, c_arr, eta, projection)
            v_batch = np.asarray(c_arr.reshape(-1))[:, None] * np.asarray(k_arr @ np.asarray(projection).T)
            err = np.asarray(k_arr @ before.T - v_batch)
            grad = (err.T @ np.asarray(k_arr)) / batch_size
            grad_norm = float(np.linalg.norm(grad))
        new_matrices.append(
            assert_finite_array(
                updated,
                level=level_idx,
                global_step=step,
                eta=float(eta),
                batch_size=batch_size,
                matrix_norm=float(np.linalg.norm(before)),
                gradient_norm=grad_norm,
                name=f"correctness M^{level_idx} after update",
            )
        )

    new_state = CorrectnessMemoryState(
        matrices=tuple(new_matrices),
        step=jnp.array(step, dtype=jnp.int32),
    )
    return _assert_finite_correctness_state(new_state, global_step=step, batch_size=batch_size)


def _assert_finite_correctness_state(
    state: CorrectnessMemoryState,
    *,
    global_step: int | None = None,
    eta: float | None = None,
    batch_size: int | None = None,
) -> CorrectnessMemoryState:
    checked = []
    for level_idx, matrix in enumerate(state.matrices, start=1):
        arr = np.asarray(matrix, dtype=np.float32)
        checked.append(
            assert_finite_array(
                arr,
                level=level_idx,
                global_step=global_step if global_step is not None else int(state.step),
                eta=eta,
                batch_size=batch_size,
                matrix_norm=float(np.linalg.norm(arr)),
                name=f"correctness M^{level_idx}",
            )
        )
    return CorrectnessMemoryState(matrices=tuple(checked), step=state.step)


class CorrectnessMemory:
    """Stateful wrapper for correctness memory with configurable value dim d_v."""

    def __init__(self, d_k: int, cfg: AssociativeMemoryConfig | None = None) -> None:
        self.d_k = int(d_k)
        self.cfg = cfg or AssociativeMemoryConfig()
        self.d_v = int(self.cfg.correctness_z_dim)
        self.projection = correctness_value_projection_matrix(
            self.d_k,
            self.d_v,
            seed=self.cfg.aggregation_seed,
        )
        self._state = init_correctness_memory(d_k, self.cfg)

    @property
    def state(self) -> CorrectnessMemoryState:
        return self._state

    def reset(self) -> None:
        self._state = init_correctness_memory(self.d_k, self.cfg)

    def retrieve(self, k: jnp.ndarray) -> jnp.ndarray:
        return retrieve_all_correctness_levels(k, self._state)

    def retrieve_level(self, k: jnp.ndarray, level: int) -> jnp.ndarray:
        return correctness_retrieve(k, self._state.matrices[level])

    def update(self, k: jnp.ndarray, c: jnp.ndarray) -> CorrectnessMemoryState:
        self._state = correctness_update(self._state, k, c, self.cfg, d_k=self.d_k)
        return self._state


__all__ = [
    "CorrectnessMemory",
    "CorrectnessMemoryState",
    "batch_correctness_delta_rule_step",
    "correctness_delta_rule_step",
    "correctness_from_logits",
    "correctness_retrieve",
    "correctness_update",
    "correctness_value_projection_matrix",
    "correctness_value_vector",
    "expected_matrix_shape",
    "init_correctness_memory",
    "normalize_key",
    "resolve_correctness_dims",
    "retrieve_all_correctness_levels",
]
