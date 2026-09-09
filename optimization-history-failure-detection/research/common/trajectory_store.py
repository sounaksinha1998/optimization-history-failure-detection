"""Shared-grid associative memory checkpoints and frozen-query trajectories.

Training-time construction
--------------------------
At global step t, the associative matrices M_t^j are updated from the *moving*
batch representation (k_t, v_t). Those updates are not stored as per-sample
visit rows.

Checkpoint grid
---------------
Every ``trajectory_sample_every`` global steps (and at the final step) the
complete state {M_t^j}_j is saved. All samples share:

    checkpoint_steps = [t_1 < t_2 < ... < t_T]

Post-training historical query
------------------------------
After the classifier is frozen, each sample x has one query:

    k_query(x) = normalize(h_T(x))

Replay does not use h_t:

    R[t, j, :](x) = M_t^j @ k_query(x)

so R(x) ∈ R^{T × K × d_v} with the same T and K for every split.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from optimizer.associative_memory import (
    AssociativeMemoryConfig,
    AssociativeMemoryState,
    assert_finite_array,
    assert_finite_state,
    associative_retrieve,
    normalize_key,
    randomize_associative_memory,
)


class MissingMemoryCheckpointsError(RuntimeError):
    """Raised when evaluation is attempted without a shared checkpoint grid."""


@dataclass(frozen=True)
class MemoryCheckpointBundle:
    """Complete associative states on a common global-step grid.

    ``matrices`` has shape (T, K, d_v, d_k) with T = len(checkpoint_steps).
    """

    checkpoint_steps: np.ndarray
    matrices: np.ndarray
    num_levels: int
    d_v: int
    d_k: int
    sample_every: int

    @property
    def T(self) -> int:
        return int(self.checkpoint_steps.shape[0])

    def state_at(self, time_index: int) -> AssociativeMemoryState:
        mats = tuple(jnp.asarray(m, dtype=jnp.float32) for m in self.matrices[time_index])
        step = int(self.checkpoint_steps[time_index])
        return AssociativeMemoryState(matrices=mats, step=jnp.array(step, dtype=jnp.int32))


def assert_checkpoint_consistency(bundle: MemoryCheckpointBundle) -> None:
    """Identical K, identical matrix shapes, strictly increasing global steps."""
    steps = np.asarray(bundle.checkpoint_steps, dtype=np.int64)
    mats = np.asarray(bundle.matrices, dtype=np.float32)
    if steps.ndim != 1 or steps.size == 0:
        raise MissingMemoryCheckpointsError(
            f"checkpoint_steps must be a non-empty 1-D grid, got shape {steps.shape}"
        )
    if mats.ndim != 4:
        raise ValueError(f"matrices must have shape (T, K, d_v, d_k), got {mats.shape}")
    t_steps, k_levels, d_v, d_k = mats.shape
    if t_steps != steps.shape[0]:
        raise ValueError(f"T mismatch: {t_steps} matrices vs {steps.shape[0]} steps")
    if k_levels != bundle.num_levels or d_v != bundle.d_v or d_k != bundle.d_k:
        raise ValueError("bundle metadata does not match matrices shape")
    if np.any(np.diff(steps) <= 0):
        raise ValueError(f"checkpoint_steps must be strictly increasing, got {steps.tolist()}")
    for t in range(t_steps):
        for level in range(k_levels):
            assert_finite_array(
                mats[t, level],
                level=level + 1,
                global_step=int(steps[t]),
                name=f"checkpoint M^{level + 1}",
            )


def assert_trajectory_consistency(R: np.ndarray, *, T: int, K: int) -> None:
    """Every sample has identical time length T and level count K."""
    arr = np.asarray(R, dtype=np.float32)
    if arr.ndim == 3:
        t_steps, k_levels, _d_v = arr.shape
        if t_steps != T or k_levels != K:
            raise ValueError(f"trajectory shape {arr.shape} != (T={T}, K={K}, d_v)")
        return
    if arr.ndim != 4:
        raise ValueError(f"R must be (T, K, d_v) or (N, T, K, d_v), got {arr.shape}")
    _n, t_steps, k_levels, _d_v = arr.shape
    if t_steps != T or k_levels != K:
        raise ValueError(f"trajectory shape {arr.shape} != (N, T={T}, K={K}, d_v)")


class MemoryCheckpointStore:
    """Accumulate complete M_t states on a shared global training-time grid."""

    def __init__(self, *, sample_every: int = 1) -> None:
        if sample_every <= 0:
            raise ValueError(f"sample_every must be positive, got {sample_every}")
        self.sample_every = int(sample_every)
        self._frozen = False
        self._steps: list[int] = []
        self._matrices: list[np.ndarray] = []

    @classmethod
    def from_config(cls, cfg: AssociativeMemoryConfig) -> MemoryCheckpointStore:
        return cls(sample_every=cfg.trajectory_sample_every)

    @property
    def frozen(self) -> bool:
        return self._frozen

    def should_record(self, step: int) -> bool:
        return int(step) > 0 and int(step) % self.sample_every == 0

    def save_memory_checkpoint(self, global_step: int, memory_states: AssociativeMemoryState) -> None:
        """Save {M_t^j} at ``global_step`` when the step lands on the shared grid."""
        if self._frozen:
            raise RuntimeError("cannot save_memory_checkpoint on a frozen store")
        step = int(global_step)
        if step <= 0:
            raise ValueError(f"global_step must be positive, got {step}")
        if not self.should_record(step):
            return
        if self._steps and step <= self._steps[-1]:
            raise ValueError(
                f"checkpoint global_step must increase (last={self._steps[-1]}, got={step})"
            )
        state = assert_finite_state(memory_states, global_step=step)
        stacked = np.stack([np.asarray(m, dtype=np.float32) for m in state.matrices], axis=0)
        self._steps.append(step)
        self._matrices.append(stacked)

    def save_final_checkpoint(self, global_step: int, memory_states: AssociativeMemoryState) -> None:
        """Record the last training step even if it is off the regular grid."""
        if self._frozen:
            raise RuntimeError("cannot save_final_checkpoint on a frozen store")
        step = int(global_step)
        if self._steps and step == self._steps[-1]:
            return
        if self._steps and step < self._steps[-1]:
            raise ValueError(f"final step {step} precedes last checkpoint {self._steps[-1]}")
        state = assert_finite_state(memory_states, global_step=step)
        stacked = np.stack([np.asarray(m, dtype=np.float32) for m in state.matrices], axis=0)
        self._steps.append(step)
        self._matrices.append(stacked)

    def freeze(self) -> MemoryCheckpointBundle:
        if not self._steps:
            raise MissingMemoryCheckpointsError(
                "No associative memory checkpoints were saved. Evaluation cannot "
                "fall back to T=1; save M_t on the shared global-step grid during training."
            )
        matrices = np.stack(self._matrices, axis=0)
        bundle = MemoryCheckpointBundle(
            checkpoint_steps=np.asarray(self._steps, dtype=np.int32),
            matrices=matrices,
            num_levels=int(matrices.shape[1]),
            d_v=int(matrices.shape[2]),
            d_k=int(matrices.shape[3]),
            sample_every=self.sample_every,
        )
        assert_checkpoint_consistency(bundle)
        self._frozen = True
        return bundle


def build_query_trajectory(
    h_final: np.ndarray | jnp.ndarray,
    memory_checkpoints: MemoryCheckpointBundle,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """Replay frozen k_query = normalize(h_T) against every saved M_t.

    Parameters
    ----------
    h_final
        Frozen penultimate features, shape (d_h,) or (N, d_h). This must be h_T,
        not a historical h_t.
    memory_checkpoints
        Shared-grid {M_t^j}.

    Returns
    -------
    np.ndarray
        R with shape (T, K, d_v) or (N, T, K, d_v).
    """
    assert_checkpoint_consistency(memory_checkpoints)
    h_arr = np.asarray(h_final, dtype=np.float32)
    if h_arr.ndim == 1:
        h_arr = h_arr[None, :]
        squeeze = True
    elif h_arr.ndim == 2:
        squeeze = False
    else:
        raise ValueError(f"h_final must have shape (d_h,) or (N, d_h), got {h_arr.shape}")

    k_query = np.asarray(normalize_key(jnp.asarray(h_arr), eps=eps))
    t_steps = memory_checkpoints.T
    k_levels = memory_checkpoints.num_levels
    d_v = memory_checkpoints.d_v
    n_samples = k_query.shape[0]
    R = np.zeros((n_samples, t_steps, k_levels, d_v), dtype=np.float32)
    for t in range(t_steps):
        state = memory_checkpoints.state_at(t)
        for level, matrix in enumerate(state.matrices):
            R[:, t, level, :] = np.asarray(associative_retrieve(jnp.asarray(k_query), matrix))
    assert_trajectory_consistency(R, T=t_steps, K=k_levels)
    if squeeze:
        return R[0]
    return R


def randomize_checkpoint_bundle(
    bundle: MemoryCheckpointBundle,
    seed: int,
) -> MemoryCheckpointBundle:
    """Norm-matched random {M_t^j} on the same checkpoint grid."""
    assert_checkpoint_consistency(bundle)
    mats = []
    for t in range(bundle.T):
        rand_state = randomize_associative_memory(bundle.state_at(t), seed=seed + 10_003 * t)
        mats.append(np.stack([np.asarray(m, dtype=np.float32) for m in rand_state.matrices], axis=0))
    return MemoryCheckpointBundle(
        checkpoint_steps=np.asarray(bundle.checkpoint_steps, dtype=np.int32),
        matrices=np.stack(mats, axis=0),
        num_levels=bundle.num_levels,
        d_v=bundle.d_v,
        d_k=bundle.d_k,
        sample_every=bundle.sample_every,
    )


def save_checkpoint_bundle(
    path: Path,
    bundle: MemoryCheckpointBundle,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "checkpoint_steps": np.asarray(bundle.checkpoint_steps, dtype=np.int32),
        "matrices": np.asarray(bundle.matrices, dtype=np.float32),
        "num_levels": np.array(bundle.num_levels, dtype=np.int32),
        "d_v": np.array(bundle.d_v, dtype=np.int32),
        "d_k": np.array(bundle.d_k, dtype=np.int32),
        "sample_every": np.array(bundle.sample_every, dtype=np.int32),
    }
    if metadata is not None:
        payload["metadata_json"] = np.array(json.dumps(metadata))
    np.savez_compressed(path, **payload)


def load_checkpoint_bundle(path: Path) -> MemoryCheckpointBundle:
    data = np.load(Path(path), allow_pickle=True)
    bundle = MemoryCheckpointBundle(
        checkpoint_steps=np.asarray(data["checkpoint_steps"], dtype=np.int32),
        matrices=np.asarray(data["matrices"], dtype=np.float32),
        num_levels=int(data["num_levels"]),
        d_v=int(data["d_v"]),
        d_k=int(data["d_k"]),
        sample_every=int(data["sample_every"]),
    )
    assert_checkpoint_consistency(bundle)
    return bundle


# Backward-compatible names used by older tests/docs. These are the checkpoint
# APIs, not per-sample visit trajectories.
FrozenTrajectoryBundle = MemoryCheckpointBundle
save_trajectory_bundle = save_checkpoint_bundle
load_trajectory_bundle = load_checkpoint_bundle
