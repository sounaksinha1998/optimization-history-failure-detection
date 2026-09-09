"""Deployment-time inference for historical associative memory.

Label-free pipeline (plan: deployment-pipeline.plan.md):

    x → h(x) → k(x) → M^(j) k(x) = z_j(x)

Optional historical checkpoints yield z_{t,j}(x) = M_t^(j) k(x).

This module does **not** compute v = y - p, deferral scores, or any final
routing policy. Ground truth is joined only in offline evaluation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from optimizer.associative_memory import (
    AssociativeMemoryConfig,
    AssociativeMemoryState,
    associative_retrieve,
    normalize_key,
    retrieve_all_levels,
)
from research.common.resnet import ResNetParams, resnet18_features, resnet18_probs
from research.common.trajectory_store import MemoryCheckpointBundle

SplitName = Literal["train", "cal", "test", "external"]


@dataclass(frozen=True)
class PredictionOutput:
    """Ordinary classifier output at deployment."""

    probabilities: np.ndarray
    predicted_class: int
    entropy: float
    normalized_entropy: float


@dataclass(frozen=True)
class RepresentationOutput:
    """Penultimate representation and normalized query key."""

    h: np.ndarray
    key: np.ndarray


@dataclass(frozen=True)
class LevelMemoryResponse:
    """Per-level memory response z_j(x) = M^(j) k(x)."""

    level: int
    response: np.ndarray
    magnitude: float
    direction: np.ndarray


@dataclass(frozen=True)
class CrossLevelDiagnostics:
    """Cross-level agreement and variance diagnostics."""

    pairwise_cosine: np.ndarray
    response_variance: np.ndarray
    response_variance_mean: float
    magnitude_variance: float
    mean_pairwise_cosine: float


@dataclass(frozen=True)
class HistoricalCheckpointResponse:
    """Query-specific responses z_{t,j}(x) at one checkpoint time."""

    checkpoint_index: int
    checkpoint_step: int
    level_responses: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class DeploymentSampleRecord:
    """Structured per-sample deployment output (no labels)."""

    sample_id: str | int
    prediction: PredictionOutput
    representation: RepresentationOutput
    memory_levels: tuple[LevelMemoryResponse, ...]
    cross_level: CrossLevelDiagnostics
    optional_history: tuple[HistoricalCheckpointResponse, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Nested dict suitable for JSON export (arrays as lists)."""

        def _vec(arr: np.ndarray) -> list[float]:
            return np.asarray(arr, dtype=np.float32).reshape(-1).tolist()

        memory = {
            f"level_{lvl.level}": {
                "response_vector": _vec(lvl.response),
                "magnitude": lvl.magnitude,
                "normalized_direction": _vec(lvl.direction),
            }
            for lvl in self.memory_levels
        }
        history = None
        if self.optional_history is not None:
            history = []
            for snap in self.optional_history:
                history.append(
                    {
                        "checkpoint_index": snap.checkpoint_index,
                        "checkpoint_step": snap.checkpoint_step,
                        "levels": {
                            f"level_{j + 1}": _vec(z)
                            for j, z in enumerate(snap.level_responses)
                        },
                    }
                )
        return {
            "sample_id": self.sample_id,
            "prediction": {
                "probabilities": _vec(self.prediction.probabilities),
                "predicted_class": int(self.prediction.predicted_class),
                "entropy": float(self.prediction.entropy),
                "normalized_entropy": float(self.prediction.normalized_entropy),
            },
            "representation": {
                "h": _vec(self.representation.h),
                "key": _vec(self.representation.key),
            },
            "memory": {
                **memory,
                "cross_level": {
                    "pairwise_cosine_agreement": self.cross_level.pairwise_cosine.tolist(),
                    "response_variance": _vec(self.cross_level.response_variance),
                    "response_variance_mean": float(self.cross_level.response_variance_mean),
                    "magnitude_variance": float(self.cross_level.magnitude_variance),
                    "mean_pairwise_cosine": float(self.cross_level.mean_pairwise_cosine),
                },
            },
            "optional_history": history,
        }


def predictive_entropy_scalar(probs: np.ndarray, *, eps: float = 1e-12) -> float:
    """H(x) = -sum_c p_c log p_c for one probability vector."""
    p = np.clip(np.asarray(probs, dtype=np.float64).reshape(-1), eps, 1.0)
    return float(-np.sum(p * np.log(p)))


def normalized_entropy_scalar(probs: np.ndarray, num_classes: int, *, eps: float = 1e-12) -> float:
    """H_tilde(x) = H(x) / log(C)."""
    if num_classes <= 1:
        return 0.0
    return predictive_entropy_scalar(probs, eps=eps) / float(np.log(num_classes))


def compute_prediction_output(
    probs: np.ndarray,
    *,
    num_classes: int,
    eps: float = 1e-12,
) -> PredictionOutput:
    """Build prediction block from softmax probabilities."""
    p = np.asarray(probs, dtype=np.float32).reshape(-1)
    if p.shape[0] != num_classes:
        raise ValueError(f"Expected {num_classes} probabilities, got {p.shape[0]}")
    pred = int(np.argmax(p))
    ent = predictive_entropy_scalar(p, eps=eps)
    norm_ent = normalized_entropy_scalar(p, num_classes, eps=eps)
    return PredictionOutput(
        probabilities=p,
        predicted_class=pred,
        entropy=ent,
        normalized_entropy=norm_ent,
    )


def compute_representation_and_key(
    params: ResNetParams,
    x: np.ndarray,
    *,
    eps: float = 1e-8,
) -> RepresentationOutput:
    """h(x) and k(x) = h / (||h|| + eps). No labels."""
    x_j = jnp.asarray(x, dtype=jnp.float32)
    h = np.asarray(resnet18_features(params, x_j), dtype=np.float32).reshape(-1)
    k = np.asarray(normalize_key(jnp.asarray(h), eps=eps), dtype=np.float32).reshape(-1)
    return RepresentationOutput(h=h, key=k)


def query_memory_levels(
    key: np.ndarray,
    memory_state: AssociativeMemoryState,
    *,
    eps: float = 1e-8,
) -> tuple[LevelMemoryResponse, ...]:
    """z_j(x) = M^(j) k(x) for every level."""
    k_j = jnp.asarray(key, dtype=jnp.float32).reshape(-1)
    levels: list[LevelMemoryResponse] = []
    for level_idx, matrix in enumerate(memory_state.matrices, start=1):
        z = np.asarray(associative_retrieve(k_j, matrix), dtype=np.float32).reshape(-1)
        mag = float(np.linalg.norm(z))
        direction = z / (mag + eps) if mag > 0 else np.zeros_like(z)
        levels.append(
            LevelMemoryResponse(
                level=level_idx,
                response=z,
                magnitude=mag,
                direction=direction,
            )
        )
    return tuple(levels)


def compute_cross_level_diagnostics(
    levels: Sequence[LevelMemoryResponse],
) -> CrossLevelDiagnostics:
    """Pairwise cosines, V_z, V_s across memory levels."""
    if not levels:
        raise ValueError("At least one memory level is required")
    z_stack = np.stack([lvl.response for lvl in levels], axis=0)
    k_levels = z_stack.shape[0]
    pairwise = np.eye(k_levels, dtype=np.float32)
    off_diag: list[float] = []
    for i in range(k_levels):
        for j in range(i + 1, k_levels):
            zi = z_stack[i]
            zj = z_stack[j]
            denom = (np.linalg.norm(zi) + 1e-8) * (np.linalg.norm(zj) + 1e-8)
            cos_ij = float(np.dot(zi, zj) / denom)
            pairwise[i, j] = cos_ij
            pairwise[j, i] = cos_ij
            off_diag.append(cos_ij)
    mags = np.array([lvl.magnitude for lvl in levels], dtype=np.float32)
    response_var = np.var(z_stack, axis=0)
    return CrossLevelDiagnostics(
        pairwise_cosine=pairwise,
        response_variance=response_var.astype(np.float32),
        response_variance_mean=float(np.mean(response_var)),
        magnitude_variance=float(np.var(mags)),
        mean_pairwise_cosine=float(np.mean(off_diag)) if off_diag else 1.0,
    )


def query_historical_checkpoints(
    key: np.ndarray,
    checkpoints: MemoryCheckpointBundle,
) -> tuple[HistoricalCheckpointResponse, ...]:
    """z_{t,j}(x) = M_t^(j) k(x) for all checkpoint times t."""
    k_j = jnp.asarray(key, dtype=jnp.float32).reshape(-1)
    steps = np.asarray(checkpoints.checkpoint_steps, dtype=np.int64)
    history: list[HistoricalCheckpointResponse] = []
    for t_idx in range(checkpoints.T):
        state = checkpoints.state_at(t_idx)
        level_responses = tuple(
            np.asarray(associative_retrieve(k_j, matrix), dtype=np.float32).reshape(-1)
            for matrix in state.matrices
        )
        history.append(
            HistoricalCheckpointResponse(
                checkpoint_index=t_idx,
                checkpoint_step=int(steps[t_idx]),
                level_responses=level_responses,
            )
        )
    return tuple(history)


def deploy_sample(
    params: ResNetParams,
    x: np.ndarray,
    memory_state: AssociativeMemoryState,
    *,
    sample_id: str | int,
    num_classes: int,
    checkpoints: MemoryCheckpointBundle | None = None,
    include_history: bool = False,
    eps: float = 1e-8,
) -> DeploymentSampleRecord:
    """Full label-free deployment for one sample."""
    rep = compute_representation_and_key(params, x, eps=eps)
    probs = np.asarray(resnet18_probs(params, jnp.asarray(x, dtype=jnp.float32)), dtype=np.float32)
    if probs.ndim == 2:
        probs = probs[0]
    pred = compute_prediction_output(probs, num_classes=num_classes, eps=eps)
    levels = query_memory_levels(rep.key, memory_state, eps=eps)
    cross = compute_cross_level_diagnostics(levels)
    history = None
    if include_history:
        if checkpoints is None:
            raise ValueError("include_history=True requires checkpoints bundle")
        history = query_historical_checkpoints(rep.key, checkpoints)
    return DeploymentSampleRecord(
        sample_id=sample_id,
        prediction=pred,
        representation=rep,
        memory_levels=levels,
        cross_level=cross,
        optional_history=history,
    )


def deploy_batch(
    params: ResNetParams,
    x_batch: np.ndarray,
    memory_state: AssociativeMemoryState,
    *,
    sample_ids: Sequence[str | int],
    num_classes: int,
    checkpoints: MemoryCheckpointBundle | None = None,
    include_history: bool = False,
    eps: float = 1e-8,
) -> list[DeploymentSampleRecord]:
    """Label-free deployment for a batch (one record per row)."""
    x_arr = np.asarray(x_batch)
    if x_arr.ndim < 3:
        raise ValueError("x_batch must have shape (N, H, W, C) or (N, H, W)")
    if len(sample_ids) != x_arr.shape[0]:
        raise ValueError("sample_ids length must match batch size")
    return [
        deploy_sample(
            params,
            x_arr[i],
            memory_state,
            sample_id=sample_ids[i],
            num_classes=num_classes,
            checkpoints=checkpoints,
            include_history=include_history,
            eps=eps,
        )
        for i in range(x_arr.shape[0])
    ]


def records_to_dataframe(records: Sequence[DeploymentSampleRecord]) -> pd.DataFrame:
    """Flatten deployment records to a tabular form for analysis."""
    rows: list[dict[str, Any]] = []
    for rec in records:
        row: dict[str, Any] = {
            "sample_id": rec.sample_id,
            "predicted_class": rec.prediction.predicted_class,
            "entropy": rec.prediction.entropy,
            "normalized_entropy": rec.prediction.normalized_entropy,
            "key_norm": float(np.linalg.norm(rec.representation.key)),
            "cross_level_mean_cosine": rec.cross_level.mean_pairwise_cosine,
            "cross_level_magnitude_variance": rec.cross_level.magnitude_variance,
            "cross_level_response_variance_mean": rec.cross_level.response_variance_mean,
        }
        for lvl in rec.memory_levels:
            j = lvl.level
            row[f"z{j}_magnitude"] = lvl.magnitude
            for d, val in enumerate(lvl.response):
                row[f"z{j}_d{d}"] = float(val)
        rows.append(row)
    return pd.DataFrame(rows)


def save_deployment_records(
    records: Sequence[DeploymentSampleRecord],
    output_dir: Path,
    *,
    prefix: str = "deployment",
) -> dict[str, Path]:
    """Write JSON-lines structured records and a flat CSV summary."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / f"{prefix}_records.jsonl"
    csv_path = out / f"{prefix}_summary.csv"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict()) + "\n")
    records_to_dataframe(records).to_csv(csv_path, index=False)
    return {"jsonl": jsonl_path, "csv": csv_path}


def memory_diagnostic_feature_columns(df: pd.DataFrame) -> list[str]:
    """Scalar memory columns used for offline logistic evaluation."""
    cols = [
        "cross_level_mean_cosine",
        "cross_level_magnitude_variance",
        "cross_level_response_variance_mean",
    ]
    cols.extend(sorted(c for c in df.columns if c.startswith("z") and c.endswith("_magnitude")))
    return [c for c in cols if c in df.columns]


def join_ground_truth(
    deploy_df: pd.DataFrame,
    *,
    labels: np.ndarray,
    sample_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    """Attach y and error=1[y!=y_hat] for offline evaluation only."""
    out = deploy_df.copy()
    labels_arr = np.asarray(labels).reshape(-1)
    if len(labels_arr) != len(out):
        raise ValueError("labels length must match deployment dataframe rows")
    out["label"] = labels_arr.astype(int)
    out["error"] = (out["label"].to_numpy() != out["predicted_class"].to_numpy()).astype(int)
    if sample_ids is not None:
        out["sample_id"] = np.asarray(sample_ids).reshape(-1)
    return out


def _fit_logistic_auroc(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, float]:
    mask_tr = train_df[feature_cols + ["error"]].notna().all(axis=1).to_numpy()
    mask_te = test_df[feature_cols + ["error"]].notna().all(axis=1).to_numpy()
    if mask_tr.sum() < 10 or mask_te.sum() < 10:
        return {"auroc": float("nan"), "auprc": float("nan"), "n_train": int(mask_tr.sum()), "n_test": int(mask_te.sum())}
    x_tr = train_df.loc[mask_tr, feature_cols].to_numpy(dtype=np.float64)
    y_tr = train_df.loc[mask_tr, "error"].to_numpy(dtype=int)
    x_te = test_df.loc[mask_te, feature_cols].to_numpy(dtype=np.float64)
    y_te = test_df.loc[mask_te, "error"].to_numpy(dtype=int)
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "n_train": int(mask_tr.sum()), "n_test": int(mask_te.sum())}
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=0)),
        ]
    )
    pipe.fit(x_tr, y_tr)
    scores = pipe.predict_proba(x_te)[:, 1]
    return {
        "auroc": float(roc_auc_score(y_te, scores)),
        "auprc": float(average_precision_score(y_te, scores)),
        "n_train": int(mask_tr.sum()),
        "n_test": int(mask_te.sum()),
    }


def offline_evaluate_memory_information(
    cal_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare H, memory diagnostics, and combined models → error (plan §6).

    Does **not** produce a deferral policy.
    """
    rows: list[dict[str, Any]] = []
    h_only = _fit_logistic_auroc(cal_df, eval_df, ["normalized_entropy"])
    rows.append({"model": "prediction_only_H", **h_only})
    mem_cols = memory_diagnostic_feature_columns(eval_df)
    if mem_cols:
        mem_only = _fit_logistic_auroc(cal_df, eval_df, mem_cols)
        rows.append({"model": "memory_only", **mem_only})
        combined_cols = ["normalized_entropy"] + mem_cols
        combined = _fit_logistic_auroc(cal_df, eval_df, combined_cols)
        rows.append({"model": "combined_H_and_memory", **combined})
    # Univariate AUROC for interpretability (no logistic)
    for col in ["normalized_entropy"] + mem_cols:
        mask = eval_df[[col, "error"]].notna().all(axis=1).to_numpy()
        if mask.sum() < 10 or len(np.unique(eval_df.loc[mask, "error"])) < 2:
            auroc = float("nan")
            auprc = float("nan")
        else:
            y = eval_df.loc[mask, "error"].to_numpy()
            s = eval_df.loc[mask, col].to_numpy()
            auroc = float(roc_auc_score(y, s))
            auprc = float(average_precision_score(y, s))
        rows.append(
            {
                "model": f"univariate_{col}",
                "auroc": auroc,
                "auprc": auprc,
                "n_train": np.nan,
                "n_test": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_deployment_diagnostics(deploy_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate memory diagnostic statistics over a deployment split."""
    mem_cols = memory_diagnostic_feature_columns(deploy_df)
    stats_rows: list[dict[str, Any]] = []
    for col in ["normalized_entropy"] + mem_cols:
        if col not in deploy_df.columns:
            continue
        s = deploy_df[col].astype(float)
        stats_rows.append(
            {
                "feature": col,
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max()),
            }
        )
    return pd.DataFrame(stats_rows)


def plot_deployment_diagnostics(
    deploy_df: pd.DataFrame,
    output_dir: Path,
    *,
    prefix: str = "deployment",
    history_records: Sequence[DeploymentSampleRecord] | None = None,
    max_trajectory_samples: int = 8,
) -> dict[str, Path]:
    """Visualize entropy, magnitudes, agreement, trajectories, and errors."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    if "error" in deploy_df.columns and "normalized_entropy" in deploy_df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        err = deploy_df["error"].astype(int) == 1
        ax.scatter(
            deploy_df.loc[~err, "normalized_entropy"],
            deploy_df.loc[~err, "z1_magnitude"] if "z1_magnitude" in deploy_df else deploy_df["cross_level_mean_cosine"],
            s=8,
            alpha=0.35,
            label="correct",
        )
        ax.scatter(
            deploy_df.loc[err, "normalized_entropy"],
            deploy_df.loc[err, "z1_magnitude"] if "z1_magnitude" in deploy_df else deploy_df["cross_level_mean_cosine"],
            s=8,
            alpha=0.5,
            label="error",
        )
        ylabel = "||z_1||" if "z1_magnitude" in deploy_df.columns else "mean pairwise cos"
        ax.set_xlabel("normalized entropy H")
        ax.set_ylabel(ylabel)
        ax.set_title("Prediction uncertainty vs memory response")
        ax.legend(loc="best", fontsize=8)
        p = out / f"{prefix}_entropy_vs_memory.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths["entropy_vs_memory"] = p

    mag_cols = sorted(c for c in deploy_df.columns if c.endswith("_magnitude") and c.startswith("z"))
    if mag_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        data = [deploy_df[c].astype(float).to_numpy() for c in mag_cols]
        ax.boxplot(data, tick_labels=[c.replace("_magnitude", "") for c in mag_cols])
        ax.set_ylabel("||z_j||")
        ax.set_title("Memory response magnitudes by level")
        p = out / f"{prefix}_magnitude_by_level.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths["magnitude_by_level"] = p

    if "cross_level_mean_cosine" in deploy_df.columns and "error" in deploy_df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        err = deploy_df["error"].astype(int) == 1
        ax.hist(deploy_df.loc[~err, "cross_level_mean_cosine"], bins=30, alpha=0.6, label="correct", density=True)
        ax.hist(deploy_df.loc[err, "cross_level_mean_cosine"], bins=30, alpha=0.6, label="error", density=True)
        ax.set_xlabel("mean cross-level cosine")
        ax.set_title("Cross-level agreement vs error")
        ax.legend(loc="best", fontsize=8)
        p = out / f"{prefix}_cross_level_agreement.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths["cross_level_agreement"] = p

    if history_records:
        fig, ax = plt.subplots(figsize=(7, 4))
        shown = 0
        for rec in history_records:
            if rec.optional_history is None or shown >= max_trajectory_samples:
                break
            t_steps = [h.checkpoint_step for h in rec.optional_history]
            for lvl in range(len(rec.memory_levels)):
                mags = [
                    float(np.linalg.norm(snap.level_responses[lvl]))
                    for snap in rec.optional_history
                ]
                ax.plot(t_steps, mags, alpha=0.7, linewidth=1.0, label=f"id={rec.sample_id} L{lvl + 1}")
            shown += 1
        if shown:
            ax.set_xlabel("checkpoint global step")
            ax.set_ylabel("||z_{t,j}||")
            ax.set_title("Historical memory-response trajectories (subset)")
            if shown == 1:
                ax.legend(fontsize=6, loc="best")
            p = out / f"{prefix}_temporal_trajectories.png"
            fig.tight_layout()
            fig.savefig(p, dpi=120)
            plt.close(fig)
            paths["temporal_trajectories"] = p
        else:
            plt.close(fig)

    return paths


@dataclass
class DeploymentSanityReport:
    """Results of plan §8 sanity checks."""

    passed: bool
    checks: dict[str, bool]
    details: dict[str, Any] = field(default_factory=dict)


def run_deployment_sanity_checks(
    params: ResNetParams,
    x_batch: np.ndarray,
    memory_state: AssociativeMemoryState,
    *,
    num_classes: int,
    checkpoints: MemoryCheckpointBundle | None = None,
    eps: float = 1e-8,
    key_norm_tol: float = 1e-3,
) -> DeploymentSanityReport:
    """Verify shapes, key normalization, batch/single agreement, manual matmul."""
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    x_arr = np.asarray(x_batch)
    if x_arr.ndim == 3:
        x_arr = x_arr[None, ...]
    batch_size = x_arr.shape[0]
    sample_ids = list(range(batch_size))

    batch_records = deploy_batch(
        params,
        x_arr,
        memory_state,
        sample_ids=sample_ids,
        num_classes=num_classes,
        checkpoints=checkpoints,
        include_history=checkpoints is not None,
        eps=eps,
    )
    single_records = [
        deploy_sample(
            params,
            x_arr[i],
            memory_state,
            sample_id=i,
            num_classes=num_classes,
            checkpoints=checkpoints,
            include_history=checkpoints is not None,
            eps=eps,
        )
        for i in range(batch_size)
    ]

    # Key and matrix shapes
    rep0 = batch_records[0].representation
    d_k = rep0.key.shape[0]
    checks["key_dim"] = d_k > 0
    matrix_shapes_ok = True
    for j, matrix in enumerate(memory_state.matrices, start=1):
        m = np.asarray(matrix)
        ok = m.shape == (num_classes, d_k)
        matrix_shapes_ok = matrix_shapes_ok and ok
        z = batch_records[0].memory_levels[j - 1].response
        checks[f"response_dim_L{j}"] = z.shape == (num_classes,)
    checks["matrix_shapes"] = matrix_shapes_ok
    details["d_k"] = d_k
    details["d_v"] = num_classes
    details["num_levels"] = len(memory_state.matrices)

    # Key unit norm
    key_norms = [float(np.linalg.norm(r.representation.key)) for r in batch_records]
    checks["key_unit_norm"] = all(abs(n - 1.0) <= key_norm_tol for n in key_norms)
    details["key_norms"] = key_norms

    # Batch vs single agreement
    batch_single_ok = True
    for b, s in zip(batch_records, single_records, strict=True):
        for lb, ls in zip(b.memory_levels, s.memory_levels, strict=True):
            if not np.allclose(lb.response, ls.response, rtol=1e-5, atol=1e-6):
                batch_single_ok = False
    checks["batch_matches_single"] = batch_single_ok

    # Manual M k matches implementation
    manual_ok = True
    k0 = jnp.asarray(batch_records[0].representation.key)
    for j, matrix in enumerate(memory_state.matrices, start=1):
        manual = np.asarray(matrix @ k0, dtype=np.float32)
        impl = batch_records[0].memory_levels[j - 1].response
        if not np.allclose(manual, impl, rtol=1e-5, atol=1e-6):
            manual_ok = False
    checks["manual_matmul_matches"] = manual_ok

    # retrieve_all_levels consistency
    all_levels = np.asarray(retrieve_all_levels(k0, memory_state))
    stack_ok = all(
        np.allclose(all_levels[j], batch_records[0].memory_levels[j].response, rtol=1e-5, atol=1e-6)
        for j in range(all_levels.shape[0])
    )
    checks["retrieve_all_levels_matches"] = stack_ok

    # Near-zero response direction safeguard
    zero_dir_ok = True
    for rec in batch_records:
        for lvl in rec.memory_levels:
            if lvl.magnitude <= eps:
                if not np.allclose(lvl.direction, 0.0, atol=1e-6):
                    zero_dir_ok = False
    checks["zero_magnitude_direction_safe"] = zero_dir_ok

    # Historical checkpoints if provided
    if checkpoints is not None:
        hist = batch_records[0].optional_history
        checks["history_length"] = hist is not None and len(hist) == checkpoints.T
        details["checkpoint_T"] = checkpoints.T

    passed = all(checks.values())
    return DeploymentSanityReport(passed=passed, checks=checks, details=details)


def run_deployment_on_split(
    params: ResNetParams,
    x: np.ndarray,
    y: np.ndarray | None,
    sample_ids: np.ndarray,
    memory_state: AssociativeMemoryState,
    checkpoints: MemoryCheckpointBundle,
    *,
    num_classes: int,
    include_history: bool = True,
    eps: float = 1e-8,
) -> tuple[list[DeploymentSampleRecord], pd.DataFrame]:
    """Deploy on one data split; optionally join labels for offline eval."""
    ids = [str(s) for s in np.asarray(sample_ids).reshape(-1)]
    records = deploy_batch(
        params,
        np.asarray(x),
        memory_state,
        sample_ids=ids,
        num_classes=num_classes,
        checkpoints=checkpoints if include_history else None,
        include_history=include_history,
        eps=eps,
    )
    df = records_to_dataframe(records)
    if y is not None:
        df = join_ground_truth(df, labels=y, sample_ids=ids)
    return records, df
