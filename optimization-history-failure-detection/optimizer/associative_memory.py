"""Associative memory with K independent per-level matrices.

This module is the NEW learned key→value memory. It is not the old NRMv2
hierarchical EMA of gradients (`optimizer/memory.py`). The two must not be mixed.

Each level j maintains a matrix M^j ∈ R^{d_v × d_k} and retrieves

    L^j(k) = M^j k

Training-time construction (moving representation, allowed):

    k_t = normalize(h_t(x_t))
    v_t = y_t - p_t
    M_{t+1}^j = M_t^j - η_j (M_t^j k_t - v_t) k_t^T

Post-training historical query (frozen representation, required for R(x)):

    k_query(x) = normalize(h_T(x))
    R[t, j, :](x) = M_t^j k_query(x)

Levels use independent update schedules (no EMA nesting between levels):

| Level | Update every | η   |
|-------|-------------|-----|
| L¹    | 1 step      | 0.1 |
| L²    | 4 steps     | 0.05|
| L³    | 16 steps    | 0.01|
| L⁴    | 64 steps    | 0.005|
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

DEFAULT_UPDATE_EVERY: tuple[int, ...] = (1, 4, 16, 64)
DEFAULT_LEARNING_RATES: tuple[float, ...] = (0.1, 0.05, 0.01, 0.005)


class AssociativeMemoryNaNError(RuntimeError):
    """Raised when an associative update or checkpoint contains NaN/Inf."""


@dataclass(frozen=True)
class AssociativeMemoryConfig:
    """Modular toggles for the associative memory pipeline.

    ``use_attention`` defaults to False (Option A): z_memory is a documented
    deterministic mean-pool of the spectral/raw trajectory. Learned attention
    is not part of the primary correction experiment.
    """

    use_associative_memory: bool = True
    use_fft: bool = True
    use_attention: bool = False
    trajectory_sample_every: int = 1
    num_levels: int = 4
    update_every: tuple[int, ...] = DEFAULT_UPDATE_EVERY
    learning_rates: tuple[float, ...] = DEFAULT_LEARNING_RATES
    eps: float = 1e-8
    aggregation_seed: int = 0

    def __post_init__(self) -> None:
        if self.num_levels <= 0:
            raise ValueError(f"num_levels must be positive, got {self.num_levels}")
        if len(self.update_every) < self.num_levels:
            raise ValueError(
                f"update_every has {len(self.update_every)} entries but num_levels={self.num_levels}"
            )
        if len(self.learning_rates) < self.num_levels:
            raise ValueError(
                f"learning_rates has {len(self.learning_rates)} entries but num_levels={self.num_levels}"
            )
        if self.trajectory_sample_every <= 0:
            raise ValueError(
                f"trajectory_sample_every must be positive, got {self.trajectory_sample_every}"
            )


class AssociativeMemoryState(NamedTuple):
    """Frozen-friendly state: one (d_v, d_k) matrix per level."""

    matrices: tuple[jnp.ndarray, ...]
    step: jnp.ndarray


def assert_finite_array(
    arr: np.ndarray | jnp.ndarray,
    *,
    level: int | None = None,
    global_step: int | None = None,
    eta: float | None = None,
    batch_size: int | None = None,
    matrix_norm: float | None = None,
    gradient_norm: float | None = None,
    name: str = "array",
) -> jnp.ndarray:
    """Return ``arr`` as float32, or fail loudly if any value is non-finite."""
    x = jnp.asarray(arr, dtype=jnp.float32)
    if bool(jnp.all(jnp.isfinite(x))):
        return x
    n_nan = int(np.sum(~np.isfinite(np.asarray(x))))
    parts = [
        f"Non-finite values in associative {name} (count={n_nan})",
        f"level={level}",
        f"global_step={global_step}",
        f"eta={eta}",
        f"batch_size={batch_size}",
        f"matrix_norm={matrix_norm}",
        f"gradient_norm={gradient_norm}",
    ]
    raise AssociativeMemoryNaNError("; ".join(parts))


def assert_finite_state(
    state: AssociativeMemoryState,
    *,
    global_step: int | None = None,
    eta: float | None = None,
    batch_size: int | None = None,
) -> AssociativeMemoryState:
    """Fail if any memory matrix is non-finite. Does not replace NaNs."""
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
                name=f"M^{level_idx}",
            )
        )
    return AssociativeMemoryState(matrices=tuple(checked), step=state.step)


def resolve_level_schedule(cfg: AssociativeMemoryConfig) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Return (update_every, learning_rates) truncated to num_levels."""
    return (
        tuple(int(x) for x in cfg.update_every[: cfg.num_levels]),
        tuple(float(x) for x in cfg.learning_rates[: cfg.num_levels]),
    )


def init_associative_memory(
    d_k: int,
    d_v: int,
    cfg: AssociativeMemoryConfig | None = None,
) -> AssociativeMemoryState:
    """Initialize all M^j to zeros."""
    cfg = cfg or AssociativeMemoryConfig()
    matrices = tuple(jnp.zeros((d_v, d_k), dtype=jnp.float32) for _ in range(cfg.num_levels))
    return AssociativeMemoryState(matrices=matrices, step=jnp.array(0, dtype=jnp.int32))


def normalize_key(k: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """L2-normalize keys along the last axis.

    Accepts shape (d_k,) or (batch, d_k).
    """
    k = jnp.asarray(k, dtype=jnp.float32)
    if k.ndim == 1:
        return k / (jnp.linalg.norm(k) + eps)
    norms = jnp.linalg.norm(k, axis=-1, keepdims=True)
    return k / (norms + eps)


def associative_retrieve(k: jnp.ndarray, matrix: jnp.ndarray) -> jnp.ndarray:
    """L(k) = M k for a single level.

    Parameters
    ----------
    k : (d_k,) or (batch, d_k)
    matrix : (d_v, d_k)

    Returns
    -------
    (d_v,) or (batch, d_v)
    """
    k = jnp.asarray(k, dtype=jnp.float32)
    matrix = jnp.asarray(matrix, dtype=jnp.float32)
    if k.ndim == 1:
        return matrix @ k
    return k @ matrix.T


def retrieve_all_levels(
    k: jnp.ndarray,
    state: AssociativeMemoryState,
) -> jnp.ndarray:
    """Retrieve L^j(k) for all levels.

    Returns shape (num_levels, d_v) for vector k, or (batch, num_levels, d_v) for batched k.
    """
    outputs = [associative_retrieve(k, m) for m in state.matrices]
    stacked = jnp.stack(outputs, axis=0 if k.ndim == 1 else 1)
    return assert_finite_array(stacked, name="retrieve_all_levels")


def delta_rule_step(
    matrix: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    eta: float,
) -> jnp.ndarray:
    """Single (k, v) delta-rule update: M' = M - η (M k - v) k^T."""
    k = jnp.asarray(k, dtype=jnp.float32).reshape(-1)
    v = jnp.asarray(v, dtype=jnp.float32).reshape(-1)
    pred = matrix @ k
    err = pred - v
    return matrix - jnp.asarray(eta, dtype=matrix.dtype) * jnp.outer(err, k)


def batch_delta_rule_step(
    matrix: jnp.ndarray,
    k_batch: jnp.ndarray,
    v_batch: jnp.ndarray,
    eta: float,
) -> jnp.ndarray:
    """Mean-aggregated delta rule: M' = M - η * (1/B) Σ_i outer(M k_i - v_i, k_i).

    Averaging (not summing) keeps η independent of batch size.
    """
    k_batch = jnp.asarray(k_batch, dtype=jnp.float32)
    v_batch = jnp.asarray(v_batch, dtype=jnp.float32)
    if k_batch.ndim != 2 or v_batch.ndim != 2:
        raise ValueError("k_batch and v_batch must be 2-D (batch, dim)")
    batch_size = int(k_batch.shape[0])
    if batch_size == 0:
        raise ValueError("k_batch must be non-empty")
    pred = k_batch @ matrix.T
    err = pred - v_batch
    grad = (err.T @ k_batch) / jnp.asarray(batch_size, dtype=matrix.dtype)
    return matrix - jnp.asarray(eta, dtype=matrix.dtype) * grad


def associative_update(
    state: AssociativeMemoryState,
    k: jnp.ndarray,
    v: jnp.ndarray,
    cfg: AssociativeMemoryConfig | None = None,
) -> AssociativeMemoryState:
    """Advance associative memory by one global step.

    Parameters
    ----------
    k : normalized key(s), shape (d_k,) or (batch, d_k)
    v : value residual(s), shape (d_v,) or (batch, d_v)

    Training uses the moving representation k_t. Do not use this path to build
    evaluation trajectories; those must replay frozen k_query = normalize(h_T).
    """
    cfg = cfg or AssociativeMemoryConfig()
    update_every, learning_rates = resolve_level_schedule(cfg)
    step = int(state.step) + 1
    k_arr = jnp.asarray(k, dtype=jnp.float32)
    v_arr = jnp.asarray(v, dtype=jnp.float32)
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
            updated = delta_rule_step(matrix, k_arr, v_arr, eta)
            pred = matrix @ k_arr.reshape(-1)
            err = pred - v_arr.reshape(-1)
            grad_norm = float(np.linalg.norm(np.outer(np.asarray(err), np.asarray(k_arr.reshape(-1)))))
        else:
            updated = batch_delta_rule_step(matrix, k_arr, v_arr, eta)
            pred = k_arr @ matrix.T
            err = pred - v_arr
            grad = np.asarray((err.T @ k_arr) / batch_size)
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
                name=f"M^{level_idx} after update",
            )
        )

    new_state = AssociativeMemoryState(
        matrices=tuple(new_matrices),
        step=jnp.array(step, dtype=jnp.int32),
    )
    return assert_finite_state(new_state, global_step=step, batch_size=batch_size)


class AssociativeMemory:
    """Stateful wrapper around associative memory update and retrieval."""

    def __init__(
        self,
        d_k: int,
        d_v: int,
        cfg: AssociativeMemoryConfig | None = None,
    ) -> None:
        self.d_k = int(d_k)
        self.d_v = int(d_v)
        self.cfg = cfg or AssociativeMemoryConfig()
        self._state = init_associative_memory(d_k, d_v, self.cfg)

    @property
    def state(self) -> AssociativeMemoryState:
        return self._state

    def reset(self) -> None:
        self._state = init_associative_memory(self.d_k, self.d_v, self.cfg)

    def retrieve(self, k: jnp.ndarray) -> jnp.ndarray:
        """L^j(k) for all levels; shape (num_levels, d_v) or (batch, num_levels, d_v)."""
        return retrieve_all_levels(k, self._state)

    def retrieve_level(self, k: jnp.ndarray, level: int) -> jnp.ndarray:
        """L^j(k) for a single level (0-indexed)."""
        return associative_retrieve(k, self._state.matrices[level])

    def update(self, k: jnp.ndarray, v: jnp.ndarray) -> AssociativeMemoryState:
        """Apply delta-rule update and return new state."""
        self._state = associative_update(self._state, k, v, self.cfg)
        return self._state


def _frobenius_norm(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(matrix.reshape(-1)))


def randomize_associative_memory(
    state: AssociativeMemoryState,
    seed: int,
) -> AssociativeMemoryState:
    """Frozen random memory with per-level Frobenius norms matched to actual matrices."""
    rng = np.random.default_rng(seed)
    new_matrices: list[jnp.ndarray] = []
    for matrix in state.matrices:
        arr = np.asarray(matrix, dtype=np.float32)
        target_norm = _frobenius_norm(arr) + 1e-8
        rnd = rng.standard_normal(arr.shape).astype(np.float32)
        rnd = rnd / (_frobenius_norm(rnd) + 1e-8) * target_norm
        new_matrices.append(jnp.asarray(rnd))
    return AssociativeMemoryState(matrices=tuple(new_matrices), step=state.step)


def value_from_logits(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    num_classes: int,
) -> jnp.ndarray:
    """v = y_onehot - softmax(logits) for training-time value signal.

    Parameters
    ----------
    logits : (d_v,) or (batch, d_v)
    labels : () int or (batch,) int
    """
    logits = jnp.asarray(logits, dtype=jnp.float32)
    probs = jax_softmax(logits)
    if logits.ndim == 1:
        one_hot = jnp.zeros((num_classes,), dtype=jnp.float32).at[int(labels)].set(1.0)
        return one_hot - probs
    labels = jnp.asarray(labels, dtype=jnp.int32)
    one_hot = jax_one_hot(labels, num_classes)
    return one_hot - probs


def jax_softmax(logits: jnp.ndarray) -> jnp.ndarray:
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    exp_logits = jnp.exp(logits)
    return exp_logits / jnp.sum(exp_logits, axis=-1, keepdims=True)


def jax_one_hot(labels: jnp.ndarray, num_classes: int) -> jnp.ndarray:
    return jnp.eye(num_classes, dtype=jnp.float32)[labels]
