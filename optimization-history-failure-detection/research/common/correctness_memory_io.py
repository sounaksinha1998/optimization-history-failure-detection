"""Persistence for correctness-memory artifacts (parallel to memory.py).

Saved matrix shape per level: (d_v, d_k) with d_v = correctness_z_dim. Files use ``*_correctness_associative.npz``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from optimizer.associative_memory import AssociativeMemoryConfig
from optimizer.correctness_memory import CorrectnessMemoryState, expected_matrix_shape
from research.common.trajectory_store import MemoryCheckpointBundle, load_checkpoint_bundle, save_checkpoint_bundle


def correctness_memory_path(output_dir: Path, task: str, seed: int) -> Path:
    return Path(output_dir) / "memory" / f"{task}_seed{seed}_correctness_associative.npz"


def correctness_checkpoint_bundle_path(output_dir: Path, task: str, seed: int) -> Path:
    return Path(output_dir) / "memory" / f"{task}_seed{seed}_correctness_checkpoints.npz"


def correctness_artifacts_exist(source_dir: Path, task: str, seed: int) -> bool:
    root = Path(source_dir)
    return correctness_memory_path(root, task, seed).exists() and correctness_checkpoint_bundle_path(
        root, task, seed
    ).exists()


def correctness_memory_exists(source_dir: Path, task: str, seed: int) -> bool:
    return correctness_memory_path(Path(source_dir), task, seed).exists()


def bootstrap_classifier_checkpoint_if_missing(
    correctness_output_dir: Path,
    residual_output_dir: Path,
    task: str,
    seed: int,
) -> Path:
    """Copy classifier .pkl from residual track if correctness run saved memory but not ckpt.

    Classifier training is identical between tracks; only the memory side-channel differs.
    """
    from research.common.clinical_training import checkpoint_path

    dst = checkpoint_path(correctness_output_dir, task, seed)
    if dst.exists():
        return dst
    src = checkpoint_path(residual_output_dir, task, seed)
    if not src.exists():
        raise FileNotFoundError(
            f"Classifier checkpoint missing at {dst} and no residual copy at {src}"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    return dst


def correctness_state_summary(state: CorrectnessMemoryState) -> dict[str, float | int]:
    summary: dict[str, float | int] = {"correctness_assoc_step": int(state.step)}
    for level_idx, matrix in enumerate(state.matrices, start=1):
        arr = np.asarray(matrix, dtype=np.float32)
        summary[f"correctness_M{level_idx}_norm"] = float(np.linalg.norm(arr))
        summary[f"correctness_M{level_idx}_shape"] = f"{arr.shape[0]}x{arr.shape[1]}"
        summary[f"correctness_M{level_idx}_d_v"] = int(arr.shape[0])
        summary[f"correctness_M{level_idx}_d_k"] = int(arr.shape[1])
    return summary


def save_correctness_memory(
    path: Path,
    state: CorrectnessMemoryState,
    *,
    task: str,
    seed: int,
    cfg: AssociativeMemoryConfig | None = None,
) -> dict[str, float | int]:
    cfg = cfg or AssociativeMemoryConfig()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrices = np.stack([np.asarray(m, dtype=np.float32) for m in state.matrices], axis=0)
    d_v = int(matrices.shape[1])
    d_k = int(matrices.shape[2])
    if matrices.shape[1:] != expected_matrix_shape(d_k, d_v):
        raise ValueError(f"Invalid matrix shape {matrices.shape[1:]}")
    summary = correctness_state_summary(state)
    np.savez_compressed(
        path,
        task=np.array(task),
        seed=np.array(seed),
        matrices=matrices,
        step=np.array(int(state.step), dtype=np.int32),
        num_levels=np.array(len(state.matrices), dtype=np.int32),
        d_v=np.array(d_v, dtype=np.int32),
        d_k=np.array(d_k, dtype=np.int32),
        correctness_z_dim=np.array(d_v, dtype=np.int32),
        memory_kind=np.array("correctness_hd"),
        config_json=np.array(json.dumps(asdict(cfg))),
        summary_json=np.array(json.dumps(summary)),
    )
    return summary


def load_correctness_memory(
    path: Path,
) -> tuple[CorrectnessMemoryState, AssociativeMemoryConfig, dict[str, Any]]:
    data = np.load(Path(path), allow_pickle=False)
    matrices = tuple(jnp.asarray(m, dtype=jnp.float32) for m in data["matrices"])
    state = CorrectnessMemoryState(
        matrices=matrices,
        step=jnp.array(int(data["step"]), dtype=jnp.int32),
    )
    for matrix in state.matrices:
        arr = np.asarray(matrix, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            raise ValueError("Non-finite values in loaded correctness memory matrix")
        if arr.shape[0] == arr.shape[1] and arr.shape[0] > 32:
            raise ValueError(
                f"Legacy square d×d correctness memory (shape {arr.shape}) is unsupported. "
                "Retrain with RETRAIN=True after the z_dim=7 upgrade."
            )
    if int(state.step) < 0:
        raise ValueError("Invalid correctness memory step")
    cfg_dict = json.loads(str(data["config_json"]))
    allowed = {f.name for f in fields(AssociativeMemoryConfig)}
    cfg = AssociativeMemoryConfig(**{k: v for k, v in cfg_dict.items() if k in allowed})
    metadata = {
        "task": str(data["task"]),
        "seed": int(data["seed"]),
        "summary": json.loads(str(data["summary_json"])) if "summary_json" in data else {},
    }
    return state, cfg, metadata


def bundle_from_final_correctness_state(
    state: CorrectnessMemoryState,
    *,
    cfg: AssociativeMemoryConfig | None = None,
) -> MemoryCheckpointBundle:
    """Build a T=1 checkpoint bundle from final M_T (for deploy / recovery)."""
    cfg = cfg or AssociativeMemoryConfig()
    matrices = np.stack([np.asarray(m, dtype=np.float32) for m in state.matrices], axis=0)
    d_v = int(matrices.shape[1])
    d_k = int(matrices.shape[2])
    stacked = matrices[None, ...]
    return MemoryCheckpointBundle(
        checkpoint_steps=np.asarray([int(state.step)], dtype=np.int32),
        matrices=stacked,
        num_levels=int(matrices.shape[0]),
        d_v=d_v,
        d_k=d_k,
        sample_every=cfg.trajectory_sample_every,
    )


def recover_correctness_checkpoint_bundle(
    output_dir: Path,
    task: str,
    seed: int,
) -> Path:
    """Write missing ``*_correctness_checkpoints.npz`` from saved final memory."""
    root = Path(output_dir)
    mem_path = correctness_memory_path(root, task, seed)
    ckpt_path = correctness_checkpoint_bundle_path(root, task, seed)
    if ckpt_path.exists():
        return ckpt_path
    if not mem_path.exists():
        raise FileNotFoundError(f"Cannot recover checkpoints without {mem_path}")
    state, cfg, meta = load_correctness_memory(mem_path)
    bundle = bundle_from_final_correctness_state(state, cfg=cfg)
    save_checkpoint_bundle(
        ckpt_path,
        bundle,
        metadata={"task": task, "seed": seed, "recovered_from_final_state": True, **meta.get("summary", {})},
    )
    return ckpt_path


def load_correctness_artifacts(
    source_dir: Path,
    task: str,
    seed: int,
) -> tuple[CorrectnessMemoryState, MemoryCheckpointBundle, AssociativeMemoryConfig]:
    root = Path(source_dir)
    state, cfg, _meta = load_correctness_memory(correctness_memory_path(root, task, seed))
    ckpt_path = correctness_checkpoint_bundle_path(root, task, seed)
    if not ckpt_path.exists():
        recover_correctness_checkpoint_bundle(root, task, seed)
    bundle = load_checkpoint_bundle(ckpt_path)
    return state, bundle, cfg


def save_frozen_correctness_artifacts(
    output_dir: Path,
    *,
    task: str,
    seed: int,
    state: CorrectnessMemoryState,
    checkpoints: MemoryCheckpointBundle,
    cfg: AssociativeMemoryConfig,
) -> dict[str, float | int]:
    mem_summary = save_correctness_memory(
        correctness_memory_path(output_dir, task, seed),
        state,
        task=task,
        seed=seed,
        cfg=cfg,
    )
    save_checkpoint_bundle(
        correctness_checkpoint_bundle_path(output_dir, task, seed),
        checkpoints,
        metadata={"task": task, "seed": seed, **mem_summary},
    )
    return mem_summary
