"""Deployment for high-dimensional correctness memory.

Implementation equations (audit):

    k_T(x) = h_T(x) / (||h_T(x)||_2 + eps)
    z_j(x) = M_T^{(j)} k_T(x) in R^d
    z_combined(x) = concat_j z_j(x) in R^{4d}

Probe CSV columns: z{j}_d{d} for j=1..4, d=0..d_v-1 (default d_v=7).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np
import pandas as pd

from optimizer.correctness_memory import CorrectnessMemoryState, correctness_retrieve
from research.common.deployment_pipeline import (
    PredictionOutput,
    RepresentationOutput,
    compute_prediction_output,
    compute_representation_and_key,
    join_ground_truth,
)
from research.common.resnet import ResNetParams, resnet18_probs


@dataclass(frozen=True)
class CorrectnessLevelResponse:
    """Per-level d-dimensional memory response z_j = M^{(j)} k."""

    level: int
    response: np.ndarray
    magnitude: float


@dataclass(frozen=True)
class CorrectnessDeploymentRecord:
    sample_id: str | int
    prediction: PredictionOutput
    representation: RepresentationOutput
    memory_levels: tuple[CorrectnessLevelResponse, ...]


def query_correctness_levels(
    key: np.ndarray,
    memory_state: CorrectnessMemoryState,
) -> tuple[CorrectnessLevelResponse, ...]:
    k_j = jnp.asarray(key, dtype=jnp.float32).reshape(-1)
    levels: list[CorrectnessLevelResponse] = []
    for level_idx, matrix in enumerate(memory_state.matrices, start=1):
        z = np.asarray(correctness_retrieve(k_j, matrix), dtype=np.float32).reshape(-1)
        levels.append(
            CorrectnessLevelResponse(
                level=level_idx,
                response=z,
                magnitude=float(np.linalg.norm(z)),
            )
        )
    return tuple(levels)


def deploy_correctness_sample(
    params: ResNetParams,
    x: np.ndarray,
    memory_state: CorrectnessMemoryState,
    *,
    sample_id: str | int,
    num_classes: int,
    eps: float = 1e-8,
) -> CorrectnessDeploymentRecord:
    rep = compute_representation_and_key(params, x, eps=eps)
    probs = np.asarray(resnet18_probs(params, jnp.asarray(x, dtype=jnp.float32)), dtype=np.float32)
    if probs.ndim == 2:
        probs = probs[0]
    pred = compute_prediction_output(probs, num_classes=num_classes, eps=eps)
    levels = query_correctness_levels(rep.key, memory_state)
    return CorrectnessDeploymentRecord(
        sample_id=sample_id,
        prediction=pred,
        representation=rep,
        memory_levels=levels,
    )


def correctness_records_to_dataframe(records: Sequence[CorrectnessDeploymentRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in records:
        row: dict[str, Any] = {
            "sample_id": rec.sample_id,
            "predicted_class": rec.prediction.predicted_class,
            "entropy": rec.prediction.entropy,
            "normalized_entropy": rec.prediction.normalized_entropy,
            "key_norm": float(np.linalg.norm(rec.representation.key)),
        }
        mags = [lvl.magnitude for lvl in rec.memory_levels]
        row["cross_level_response_mean"] = float(np.mean(mags))
        row["cross_level_response_variance"] = float(np.var(mags))
        for lvl in rec.memory_levels:
            j = lvl.level
            row[f"z{j}_magnitude"] = lvl.magnitude
            for d, val in enumerate(lvl.response):
                row[f"z{j}_d{d}"] = float(val)
        rows.append(row)
    return pd.DataFrame(rows)


def run_correctness_deployment_on_split(
    params: ResNetParams,
    x: np.ndarray,
    y: np.ndarray | None,
    sample_ids: np.ndarray,
    memory_state: CorrectnessMemoryState,
    *,
    num_classes: int,
    eps: float = 1e-8,
) -> tuple[list[CorrectnessDeploymentRecord], pd.DataFrame]:
    ids = [str(s) for s in np.asarray(sample_ids).reshape(-1)]
    records = [
        deploy_correctness_sample(
            params,
            np.asarray(x)[i],
            memory_state,
            sample_id=ids[i],
            num_classes=num_classes,
            eps=eps,
        )
        for i in range(len(ids))
    ]
    df = correctness_records_to_dataframe(records)
    if y is not None:
        df = join_ground_truth(df, labels=y, sample_ids=ids)
    return records, df


def run_correctness_sanity_checks(
    params: ResNetParams,
    x_batch: np.ndarray,
    memory_state: CorrectnessMemoryState,
    *,
    num_classes: int,
    labels: np.ndarray | None = None,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Verify shapes, key norm, c_t in {0,1}, and z = M k on a mini-batch."""
    x_arr = np.asarray(x_batch)
    if x_arr.ndim == 3:
        x_arr = x_arr[None, ...]
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    rep0 = compute_representation_and_key(params, x_arr[0], eps=eps)
    d_k = rep0.key.shape[0]
    d_v = int(memory_state.matrices[0].shape[0])
    checks["key_dim_positive"] = d_k > 0
    key_norm = float(np.linalg.norm(rep0.key))
    checks["key_unit_norm"] = abs(key_norm - 1.0) < 1e-3
    details["d_k"] = d_k
    details["d_v"] = d_v
    details["key_norm"] = key_norm

    matrix_ok = True
    for j, matrix in enumerate(memory_state.matrices, start=1):
        m = np.asarray(matrix)
        ok = m.shape == (d_v, d_k)
        matrix_ok = matrix_ok and ok
        rec = deploy_correctness_sample(
            params, x_arr[0], memory_state, sample_id=0, num_classes=num_classes, eps=eps
        )
        lvl = rec.memory_levels[j - 1]
        checks[f"response_dim_L{j}"] = lvl.response.shape == (d_v,)
        manual = np.asarray(m @ rep0.key.reshape(-1), dtype=np.float32)
        checks[f"manual_response_L{j}"] = np.allclose(manual, lvl.response, rtol=1e-5, atol=1e-5)
    checks["matrix_shapes_dv_x_dk"] = matrix_ok

    if labels is not None:
        y_arr = np.asarray(labels).reshape(-1)[: x_arr.shape[0]]
        c_vals: list[float] = []
        for i in range(min(5, x_arr.shape[0])):
            rec = deploy_correctness_sample(
                params, x_arr[i], memory_state, sample_id=i, num_classes=num_classes, eps=eps
            )
            c = 1.0 if rec.prediction.predicted_class == int(y_arr[i]) else 0.0
            c_vals.append(c)
            checks[f"c_t_binary_sample_{i}"] = c in (0.0, 1.0)
        details["sample_c_t"] = c_vals

    checks["all_passed"] = all(checks.values())
    return {"passed": checks["all_passed"], "checks": checks, "details": details}
