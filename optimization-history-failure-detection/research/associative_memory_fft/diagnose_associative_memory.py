"""Cheap associative-memory diagnostics. Do not treat as a DermaMNIST result."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from optimizer.associative_memory import (  # noqa: E402
    AssociativeMemory,
    AssociativeMemoryConfig,
    normalize_key,
    value_from_logits,
)
from research.common.memory import memory_reconstruction_diagnostics  # noqa: E402
from research.common.trajectory_store import (  # noqa: E402
    MemoryCheckpointStore,
    build_query_trajectory,
)


def _synthetic_run(out_dir: Path) -> dict:
    cfg = AssociativeMemoryConfig(use_fft=True, use_attention=False, trajectory_sample_every=2)
    d_k, d_v, n_samples = 16, 7, 12
    rng = np.random.default_rng(0)
    mem = AssociativeMemory(d_k, d_v, cfg)
    store = MemoryCheckpointStore.from_config(cfg)
    norms: list[dict] = []
    nan_count = 0
    for step in range(1, 17):
        k = normalize_key(jnp.asarray(rng.standard_normal((8, d_k)), dtype=jnp.float32))
        logits = jnp.asarray(rng.standard_normal((8, d_v)), dtype=jnp.float32)
        labels = jnp.asarray(rng.integers(0, d_v, size=8), dtype=jnp.int32)
        v = value_from_logits(logits, labels, num_classes=d_v)
        mem.update(k, v)
        store.save_memory_checkpoint(step, mem.state)
        for level, matrix in enumerate(mem.state.matrices, start=1):
            arr = np.asarray(matrix)
            nan_count += int(np.sum(~np.isfinite(arr)))
            norms.append(
                {
                    "global_step": step,
                    "level": level,
                    "frobenius_norm": float(np.linalg.norm(arr)),
                }
            )
    store.save_final_checkpoint(16, mem.state)
    bundle = store.freeze()
    h_T = rng.standard_normal((n_samples, d_k)).astype(np.float32)
    k_query = np.asarray(normalize_key(jnp.asarray(h_T)))
    R = build_query_trajectory(h_T, bundle)
    assert R.shape == (n_samples, bundle.T, cfg.num_levels, d_v)
    assert np.all(np.isfinite(R))

    cal_keys = k_query[:8]
    cal_values = np.asarray(
        value_from_logits(
            jnp.asarray(rng.standard_normal((8, d_v)), dtype=jnp.float32),
            jnp.asarray(rng.integers(0, d_v, size=8), dtype=jnp.int32),
            num_classes=d_v,
        )
    )
    recon = memory_reconstruction_diagnostics(mem.state, cal_keys, cal_values)
    traj_norms = np.linalg.norm(R.reshape(n_samples, -1), axis=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(norms).to_csv(out_dir / "memory_norms.csv", index=False)
    recon.to_csv(out_dir / "reconstruction_metrics.csv", index=False)
    pd.DataFrame(
        {
            "sample_idx": np.arange(n_samples),
            "T": bundle.T,
            "K": cfg.num_levels,
            "trajectory_norm": traj_norms,
            "used_h_T": True,
        }
    ).to_csv(out_dir / "trajectory_shapes.csv", index=False)
    metadata = {
        "T": bundle.T,
        "checkpoint_steps": bundle.checkpoint_steps.tolist(),
        "K": cfg.num_levels,
        "matrix_shape": [bundle.d_v, bundle.d_k],
        "nan_inf_count": nan_count,
        "norm_mean": float(pd.DataFrame(norms)["frobenius_norm"].mean()),
        "norm_std": float(pd.DataFrame(norms)["frobenius_norm"].std()),
        "norm_max": float(pd.DataFrame(norms)["frobenius_norm"].max()),
        "trajectory_norm_mean": float(traj_norms.mean()),
        "use_fft": cfg.use_fft,
        "use_attention": cfg.use_attention,
        "attention_trained": False,
        "aggregation": "mean/raw pooling",
        "all_samples_share_T": True,
        "frozen_query": True,
        "synthetic": True,
    }
    (out_dir / "checkpoint_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out_dir / "numerical_diagnostics.json").write_text(
        json.dumps({"nan_inf_count": nan_count, "finite_R": bool(np.all(np.isfinite(R)))}, indent=2),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    out = REPO_ROOT / "research" / "associative_memory_fft" / "artifacts" / "spec_correction_diagnostics"
    metadata = _synthetic_run(out)
    print(json.dumps(metadata, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
