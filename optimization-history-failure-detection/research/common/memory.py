"""NRM v2 and associative memory diagnostics, persistence, and deploy helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from optimizer.associative_memory import (
    AssociativeMemory,
    AssociativeMemoryConfig,
    AssociativeMemoryState,
    assert_finite_state,
    associative_retrieve,
    init_associative_memory,
    normalize_key,
)
from optimizer.nrm_v2 import NRMv2State
from research.common.trajectory_store import (
    MemoryCheckpointBundle,
    load_checkpoint_bundle,
    save_checkpoint_bundle,
)


def tree_l2_norm(tree: Any) -> float:
    """Global L2 norm over all leaves in a pytree."""
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return 0.0
    return float(jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves)))


def extract_nrm_v2_state(opt_state: Any) -> NRMv2State:
    """Unwrap ``(NRMv2State, adam_state)`` from ``research_nrm_v2_observe``."""
    mem_state, _adam = opt_state
    return mem_state


extract_v5b_state = extract_nrm_v2_state  # deprecated alias


def memory_state_summary(mem_state: NRMv2State) -> dict[str, float | int]:
    """Scalar diagnostics from global NRM v2 memory at the current step."""
    summary: dict[str, float | int] = {
        "optimizer_step": int(mem_state.step),
        "mem_gamma_norm": tree_l2_norm(mem_state.gamma),
        "mem_beta_norm": tree_l2_norm(mem_state.beta),
        "mem_alpha_norm": tree_l2_norm(mem_state.alpha),
    }
    for level_idx, level in enumerate(mem_state.long_term, start=1):
        summary[f"mem_L{level_idx}_norm"] = tree_l2_norm(level)
    return summary


def long_term_depth_from_state(mem_state: NRMv2State) -> int:
    return len(mem_state.long_term)


def extract_associative_state(assoc_mem: AssociativeMemory) -> AssociativeMemoryState:
    """Return the frozen-friendly state from an :class:`AssociativeMemory` wrapper."""
    return assoc_mem.state


def associative_memory_path(output_dir: Path, task: str, seed: int) -> Path:
    return Path(output_dir) / "memory" / f"{task}_seed{seed}_associative.npz"


def checkpoint_bundle_path(output_dir: Path, task: str, seed: int) -> Path:
    return Path(output_dir) / "memory" / f"{task}_seed{seed}_checkpoints.npz"


def trajectory_path(output_dir: Path, task: str, seed: int) -> Path:
    """Path of the shared-grid checkpoint bundle (not per-sample visit rows)."""
    return checkpoint_bundle_path(output_dir, task, seed)


def associative_artifacts_exist(source_dir: Path, task: str, seed: int) -> bool:
    root = Path(source_dir)
    return associative_memory_path(root, task, seed).exists() and checkpoint_bundle_path(root, task, seed).exists()


def associative_state_summary(state: AssociativeMemoryState) -> dict[str, float | int]:
    """Scalar diagnostics from associative memory matrices."""
    summary: dict[str, float | int] = {"assoc_step": int(state.step)}
    for level_idx, matrix in enumerate(state.matrices, start=1):
        arr = np.asarray(matrix, dtype=np.float32)
        summary[f"assoc_M{level_idx}_norm"] = float(np.linalg.norm(arr))
    return summary


def save_associative_memory(
    path: Path,
    state: AssociativeMemoryState,
    *,
    task: str,
    seed: int,
    cfg: AssociativeMemoryConfig | None = None,
) -> dict[str, float | int]:
    """Persist frozen associative matrices ``M^j``."""
    cfg = cfg or AssociativeMemoryConfig()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrices = np.stack([np.asarray(m, dtype=np.float32) for m in state.matrices], axis=0)
    summary = associative_state_summary(state)
    np.savez_compressed(
        path,
        task=np.array(task),
        seed=np.array(seed),
        matrices=matrices,
        step=np.array(int(state.step), dtype=np.int32),
        num_levels=np.array(len(state.matrices), dtype=np.int32),
        d_v=np.array(matrices.shape[1], dtype=np.int32),
        d_k=np.array(matrices.shape[2], dtype=np.int32),
        config_json=np.array(json.dumps(asdict(cfg))),
        summary_json=np.array(json.dumps(summary)),
    )
    return summary


def load_associative_memory(
    path: Path,
) -> tuple[AssociativeMemoryState, AssociativeMemoryConfig, dict[str, Any]]:
    """Load frozen associative memory and its config."""
    data = np.load(Path(path), allow_pickle=False)
    matrices = tuple(jnp.asarray(m, dtype=jnp.float32) for m in data["matrices"])
    state = AssociativeMemoryState(
        matrices=matrices,
        step=jnp.array(int(data["step"]), dtype=jnp.int32),
    )
    state = assert_finite_state(state, global_step=int(state.step))
    cfg_dict = json.loads(str(data["config_json"]))
    allowed = {f.name for f in fields(AssociativeMemoryConfig)}
    cfg = AssociativeMemoryConfig(**{k: v for k, v in cfg_dict.items() if k in allowed})
    metadata = {
        "task": str(data["task"]),
        "seed": int(data["seed"]),
        "summary": json.loads(str(data["summary_json"])) if "summary_json" in data else {},
    }
    return state, cfg, metadata


def load_associative_artifacts(
    source_dir: Path,
    task: str,
    seed: int,
) -> tuple[AssociativeMemoryState, MemoryCheckpointBundle, AssociativeMemoryConfig]:
    """Load frozen final M_T and the shared-grid historical checkpoints."""
    root = Path(source_dir)
    state, cfg, _meta = load_associative_memory(associative_memory_path(root, task, seed))
    bundle = load_checkpoint_bundle(checkpoint_bundle_path(root, task, seed))
    return state, bundle, cfg


def save_frozen_associative_artifacts(
    output_dir: Path,
    *,
    task: str,
    seed: int,
    state: AssociativeMemoryState,
    checkpoints: MemoryCheckpointBundle,
    cfg: AssociativeMemoryConfig,
) -> dict[str, float | int]:
    """Save final M_T and the shared-grid checkpoint tensor."""
    mem_summary = save_associative_memory(
        associative_memory_path(output_dir, task, seed),
        state,
        task=task,
        seed=seed,
        cfg=cfg,
    )
    save_checkpoint_bundle(
        checkpoint_bundle_path(output_dir, task, seed),
        checkpoints,
        metadata={"task": task, "seed": seed, **mem_summary},
    )
    return mem_summary


def memory_reconstruction_diagnostics(
    mem_state: AssociativeMemoryState,
    keys: np.ndarray,
    values_actual: np.ndarray,
    *,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """Per-level MSE and cosine similarity between ``L^j(k)`` and ``v_actual``."""
    keys_arr = np.asarray(keys, dtype=np.float32)
    values_arr = np.asarray(values_actual, dtype=np.float32)
    if keys_arr.ndim == 1:
        keys_arr = keys_arr[None, :]
    if values_arr.ndim == 1:
        values_arr = values_arr[None, :]
    if keys_arr.shape[0] != values_arr.shape[0]:
        raise ValueError("keys and values_actual must have the same batch size")

    rows: list[dict[str, float | int]] = []
    for sample_idx in range(keys_arr.shape[0]):
        k_norm = normalize_key(jnp.asarray(keys_arr[sample_idx]), eps=eps)
        v_true = values_arr[sample_idx]
        for level_idx, matrix in enumerate(mem_state.matrices, start=1):
            v_pred = np.asarray(associative_retrieve(k_norm, matrix), dtype=np.float32)
            mse = float(np.mean((v_pred - v_true) ** 2))
            denom = (np.linalg.norm(v_pred) + eps) * (np.linalg.norm(v_true) + eps)
            cosine = float(np.dot(v_pred, v_true) / denom)
            rows.append(
                {
                    "sample_idx": sample_idx,
                    "level": level_idx,
                    "mse": mse,
                    "cosine": cosine,
                }
            )
    return pd.DataFrame(rows)


def state_from_matrices(
    matrices: np.ndarray,
    *,
    step: int = 0,
) -> AssociativeMemoryState:
    """Build :class:`AssociativeMemoryState` from a stacked ``(K, d_v, d_k)`` array."""
    mats = tuple(jnp.asarray(m, dtype=jnp.float32) for m in matrices)
    return AssociativeMemoryState(matrices=mats, step=jnp.array(step, dtype=jnp.int32))


def init_associative_from_dims(
    d_k: int,
    d_v: int,
    cfg: AssociativeMemoryConfig | None = None,
) -> AssociativeMemoryState:
    """Convenience wrapper around :func:`init_associative_memory`."""
    return init_associative_memory(d_k, d_v, cfg)
