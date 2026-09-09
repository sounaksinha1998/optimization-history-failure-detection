"""Memory identification experiments: gradient, loss, random-memory, and MHA controls.

Read-only inputs from ``research/final_experiment/``; writes only under
``research/memory_identification/artifacts/``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from optimizer.associative_memory import AssociativeMemoryConfig, randomize_associative_memory
from research.common.clinical_datasets import ClinicalDatasetConfig, load_clinical_bundle
from research.common.clinical_training import (
    ATTENTION_TAU,
    ClinicalTrainingConfig,
    checkpoint_path,
    load_checkpoint,
    save_checkpoint,
    save_frozen_memory,
    train_one_seed,
)
from research.common.error_prediction import (
    fit_failure_scorer,
    predict_error_probability,
    score_auroc_auprc,
)
from research.common.memory import (
    associative_artifacts_exist,
    extract_nrm_v2_state,
    load_associative_artifacts,
)
from research.common.msa import batch_associative_z_memory, per_sample_msa_signals
from research.common.msa_linear_deferral import (
    MSALinearDeferralConfig,
    calibrate_deferral_threshold,
    classification_metrics,
    compute_representations,
    compute_z_and_attention,
    deferral_metrics,
    deferral_probability,
    extract_split_arrays,
    fit_linear_scorers,
    fixed_memory_projection,
    load_memory_matrix_from_checkpoint,
    resolve_associative_mode,
    train_msa_on_calibration,
)
from research.common.resnet import example_grad

NUM_CLASSES_DERMA = 7
LOG_C = float(np.log(NUM_CLASSES_DERMA))
PROB_EPS = 1e-6


@dataclass
class MemoryIdentificationConfig:
    repo_root: Path | None = None
    data_dir: Path | None = None
    source_experiment_dir: Path | None = None
    output_dir: Path | None = None
    task: str = "dermamnist"
    seeds: tuple[int, ...] = (42, 123, 456)
    n_random_memory: int = 5
    random_memory_base_seed: int = 90_001
    train_if_checkpoint_missing: bool = False
    n_bootstrap: int = 2000
    bootstrap_seed: int = 20260903
    ci_level: float = 0.95
    target_deferral_rate: float = 0.20
    run_experiment1: bool = True
    run_experiment2: bool = True
    use_associative_memory: bool = False
    associative_memory: AssociativeMemoryConfig = field(default_factory=AssociativeMemoryConfig)
  # MHA settings (match msa_linear_deferral defaults)
    mha: MSALinearDeferralConfig = field(default_factory=MSALinearDeferralConfig)

    def resolve(self, cwd: Path | None = None) -> MemoryIdentificationConfig:
        root = self.repo_root
        if root is None:
            root = Path(cwd or Path.cwd()).resolve()
            if not (root / "research").exists() and (root.parent / "research").exists():
                root = root.parent
        self.repo_root = root
        if self.data_dir is None:
            self.data_dir = root / "data" / "clinical"
        if self.source_experiment_dir is None:
            self.source_experiment_dir = root / "research" / "final_experiment"
        if self.output_dir is None:
            self.output_dir = (root / "research" / "memory_identification" / "artifacts").resolve()
        else:
            self.output_dir = self.output_dir.resolve()
        self.mha.repo_root = root
        self.mha.data_dir = self.data_dir
        self.mha.source_experiment_dir = self.source_experiment_dir
        self.mha.task = self.task
        self.mha.seeds = self.seeds
        self.mha.use_associative_memory = self.use_associative_memory
        self.mha.associative_memory = self.associative_memory
        self.mha.resolve_paths(root)
        return self


def _binary_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), PROB_EPS, 1.0 - PROB_EPS)
    return float(log_loss(np.asarray(y, dtype=int), p, labels=[0, 1]))


def bootstrap_ci(values: np.ndarray, ci_level: float) -> tuple[float, float, float]:
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    a = (1.0 - ci_level) / 2.0
    return float(np.mean(v)), float(np.percentile(v, 100 * a)), float(np.percentile(v, 100 * (1 - a)))


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["error"] = (~out["correct"].astype(bool)).astype(int)
    if "normalized_entropy" not in out.columns:
        out["normalized_entropy"] = out["predictive_entropy"] / LOG_C
    out["H"] = out["normalized_entropy"].astype(float)
    out["N_actual"] = out["memory_novelty"].astype(float)
    out["G"] = out["gradient_norm"].astype(float)
    out["CE"] = out["loss"].astype(float)
    return out


def _load_split_frames(cfg: MemoryIdentificationConfig, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cal_path = cfg.source_experiment_dir / "calibration" / f"{cfg.task}_seed{seed}.csv"
    scored_path = cfg.source_experiment_dir / "scored" / f"{cfg.task}_seed{seed}.csv"
    if not cal_path.exists() or not scored_path.exists():
        raise FileNotFoundError(f"Missing calibration/scored CSV for seed {seed}")
    cal = _prepare_frame(pd.read_csv(cal_path))
    ext = _prepare_frame(pd.read_csv(scored_path))
    ext = ext[ext["split"] == "external"].reset_index(drop=True)
    return cal, ext


def _resolve_checkpoint(cfg: MemoryIdentificationConfig, seed: int) -> Path:
    local_root = cfg.output_dir / "checkpoints"
    candidates = [
        local_root / f"{cfg.task}_seed{seed}.pkl",
        cfg.source_experiment_dir / "checkpoints" / f"{cfg.task}_seed{seed}.pkl",
    ]
    for path in candidates:
        if path.exists():
            return path
    if not cfg.train_if_checkpoint_missing:
        raise FileNotFoundError(f"No checkpoint for seed {seed}; set train_if_checkpoint_missing=True")
    local_root.mkdir(parents=True, exist_ok=True)
    out_path = checkpoint_path(local_root, cfg.task, seed)  # type: ignore[arg-type]
    ds_cfg = ClinicalDatasetConfig(
        task=cfg.task,  # type: ignore[arg-type]
        data_dir=cfg.data_dir,  # type: ignore[arg-type]
        max_train=cfg.mha.max_train,
        max_cal=cfg.mha.max_cal,
        max_test=cfg.mha.max_test,
        max_external=cfg.mha.max_external,
    )
    bundle = load_clinical_bundle(ds_cfg)
    train_cfg = ClinicalTrainingConfig(
        seeds=(seed,),
        output_dir=local_root,
        epochs=8,
        batch_size=cfg.mha.batch_size,
        verbose=1,
    )
    params, opt_state, _per_sample, _metrics, summary = train_one_seed(
        bundle,
        seed=seed,
        cfg=train_cfg,
        score_splits=("cal", "test"),
    )
    save_checkpoint(
        out_path,
        params=params,
        opt_state=opt_state,
        task=cfg.task,  # type: ignore[arg-type]
        seed=seed,
        cfg=train_cfg,
        summary=summary,
    )
    mem_path = cfg.output_dir / "memory" / f"{cfg.task}_seed{seed}_memory.npz"
    save_frozen_memory(mem_path, opt_state, task=cfg.task, seed=seed)  # type: ignore[arg-type]
    return out_path


def _memory_source_dir(cfg: MemoryIdentificationConfig) -> Path:
    if (cfg.output_dir / "memory").exists():
        return cfg.output_dir  # type: ignore[return-value]
    return cfg.source_experiment_dir  # type: ignore[return-value]


def random_long_term_matched(actual_long_term: tuple[Any, ...], seed: int) -> tuple[Any, ...]:
    """Frozen random memory with per-level norms matched to actual long-term levels."""
    rng = np.random.default_rng(seed)
    out_levels: list[Any] = []
    for level in actual_long_term:
        leaves = jax.tree_util.tree_leaves(level)
        new_leaves = []
        for leaf in leaves:
            arr = np.asarray(leaf)
            flat = arr.reshape(-1)
            target_norm = float(np.linalg.norm(flat)) + 1e-8
            rnd = rng.standard_normal(flat.shape).astype(arr.dtype)
            rnd = rnd / (np.linalg.norm(rnd) + 1e-8) * target_norm
            new_leaves.append(jnp.asarray(rnd.reshape(arr.shape)))
        out_levels.append(
            jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(level), new_leaves)
        )
    return tuple(out_levels)


def random_memory_matrix_matched(M: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(M.shape[0]):
        row_norm = float(np.linalg.norm(M[i])) + 1e-8
        rnd = rng.standard_normal(M.shape[1]).astype(np.float32)
        rnd = rnd / (float(np.linalg.norm(rnd)) + 1e-8) * row_norm
        rows.append(rnd)
    return np.stack(rows, axis=0)


def randomize_trajectory_matched(R: np.ndarray, seed: int) -> np.ndarray:
    """Frozen random trajectory with per-snapshot norms matched to actual rows."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(R, dtype=np.float32)
    out = np.zeros_like(arr)
    for t in range(arr.shape[0]):
        for level in range(arr.shape[1]):
            row = arr[t, level]
            target_norm = float(np.linalg.norm(row)) + 1e-8
            rnd = rng.standard_normal(row.shape).astype(np.float32)
            rnd = rnd / (float(np.linalg.norm(rnd)) + 1e-8) * target_norm
            out[t, level] = rnd
    return out


def _z_memory_from_randomized_trajectories(
    h: np.ndarray,
    checkpoints: Any,
    *,
    seed: int,
    assoc_cfg: AssociativeMemoryConfig,
    d_out: int,
) -> np.ndarray:
    """z_random: replay the same frozen h_T against norm-matched random {M_t^j}.

    Does not depend on sample IDs. Randomization is over memory matrices on the
    shared checkpoint grid, then R[t,j] = M_rand,t^j k_query(x).
    """
    from research.common.msa import batch_associative_z_memory
    from research.common.trajectory_store import randomize_checkpoint_bundle

    rand_bundle = randomize_checkpoint_bundle(checkpoints, seed=seed)
    z, _ = batch_associative_z_memory(h, rand_bundle, assoc_cfg, d_out=d_out)
    return z


def compute_novelty_and_grad_norm(
    params: dict,
    x: np.ndarray,
    y: np.ndarray,
    long_term: tuple[Any, ...],
    *,
    attention_tau: float = ATTENTION_TAU,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    novelty = np.zeros(n, dtype=np.float64)
    grad_norm = np.zeros(n, dtype=np.float64)
    for i in range(n):
        grad = example_grad(params, jnp.asarray(x[i], dtype=jnp.float32), jnp.asarray(int(y[i])))
        signals = per_sample_msa_signals(grad, long_term, tau=attention_tau)
        novelty[i] = signals["memory_novelty"]
        leaves = jax.tree_util.tree_leaves(grad)
        flat = np.concatenate([np.asarray(leaf).reshape(-1) for leaf in leaves])
        grad_norm[i] = float(np.linalg.norm(flat))
    return novelty, grad_norm


def _fit_eval_failure_scorer(
    cal: pd.DataFrame,
    ext: pd.DataFrame,
    features: list[str],
) -> dict[str, float]:
    model = fit_failure_scorer(cal, features, error_col="error")
    scores = predict_error_probability(model, ext, features)
    y = ext["error"].to_numpy(dtype=int)
    metrics = score_auroc_auprc(y, scores)
    metrics["conditional_log_loss"] = _binary_log_loss(y, scores)
    metrics["n"] = int(len(ext))
    return metrics


def _eval_raw_feature(ext: pd.DataFrame, feature: str) -> dict[str, float]:
    y = ext["error"].to_numpy(dtype=int)
    s = ext[feature].to_numpy(dtype=float)
    metrics = score_auroc_auprc(y, s)
    metrics["conditional_log_loss"] = float("nan")
    metrics["n"] = int(len(ext))
    return metrics


def _score_split_from_checkpoint(
    params: dict,
    opt_state: Any,
    x: np.ndarray,
    y: np.ndarray,
    long_term: tuple[Any, ...],
    *,
    num_classes: int = NUM_CLASSES_DERMA,
    batch_size: int = 64,
    attention_tau: float = ATTENTION_TAU,
) -> pd.DataFrame:
    """Recompute H, CE, G, N from one frozen checkpoint (fair comparison with N_rand)."""
    from research.common.resnet import resnet18_apply, resnet18_probs

    rows: list[dict[str, float]] = []
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        xb = jnp.asarray(x[start:end], dtype=jnp.float32)
        logits = resnet18_apply(params, xb)
        probs = resnet18_probs(params, xb)
        preds = np.asarray(probs.argmax(axis=-1))
        for i in range(end - start):
            xi = xb[i]
            yi = int(y[start + i])
            grad = example_grad(params, xi, jnp.asarray(yi, dtype=jnp.int32))
            signals = per_sample_msa_signals(grad, long_term, tau=attention_tau)
            leaves = jax.tree_util.tree_leaves(grad)
            flat = np.concatenate([np.asarray(leaf).reshape(-1) for leaf in leaves])
            p = np.asarray(probs[i])
            ce = float(-np.log(max(float(p[yi]), 1e-12)))
            h_pred = float(-np.sum(p * np.log(np.clip(p, 1e-12, 1.0))))
            rows.append(
                {
                    "label": yi,
                    "prediction": int(preds[i]),
                    "correct": int(preds[i] == yi),
                    "loss": ce,
                    "H": h_pred / LOG_C,
                    "G": float(np.linalg.norm(flat)),
                    "N_actual": signals["memory_novelty"],
                }
            )
    return pd.DataFrame(rows)


def run_experiment1_seed(
    cfg: MemoryIdentificationConfig,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ckpt_path = _resolve_checkpoint(cfg, seed)
    ckpt = load_checkpoint(ckpt_path)
    params = ckpt["params"]
    mem_state = extract_nrm_v2_state(ckpt["opt_state"])
    actual_long_term = mem_state.long_term

    ds_cfg = ClinicalDatasetConfig(
        task=cfg.task,  # type: ignore[arg-type]
        data_dir=cfg.data_dir,  # type: ignore[arg-type]
        max_train=cfg.mha.max_train,
        max_cal=cfg.mha.max_cal,
        max_test=cfg.mha.max_test,
        max_external=cfg.mha.max_external,
    )
    bundle = load_clinical_bundle(ds_cfg)
    x_cal, y_cal, _ = extract_split_arrays(bundle, "calibration")
    x_ext, y_ext, _ = extract_split_arrays(bundle, "external")

    cal = _score_split_from_checkpoint(params, ckpt["opt_state"], x_cal, y_cal, actual_long_term)
    ext = _score_split_from_checkpoint(params, ckpt["opt_state"], x_ext, y_ext, actual_long_term)
    cal["error"] = (~cal["correct"].astype(bool)).astype(int)
    ext["error"] = (~ext["correct"].astype(bool)).astype(int)
    cal["CE"] = cal["loss"]
    ext["CE"] = ext["loss"]

    # Optional sanity check against final_experiment CSV when available
    try:
        csv_cal, csv_ext = _load_split_frames(cfg, seed)
        n_chk = min(8, len(ext))
        if np.allclose(ext["N_actual"].to_numpy()[:n_chk], csv_ext["N_actual"].to_numpy()[:n_chk], rtol=1e-2, atol=1e-3):
            pass
        else:
            print(
                f"[info] seed {seed}: recomputed features use checkpoint at {ckpt_path}; "
                "CSV from final_experiment may differ if checkpoints differ.",
                file=sys.stderr,
            )
    except FileNotFoundError:
        pass

    method_specs: list[tuple[str, list[str], bool, bool, str | None]] = [
        ("H", ["H"], False, False, None),
        ("N_actual", ["N_actual"], True, True, None),
        ("H+N_actual", ["H", "N_actual"], True, True, None),
        ("G", ["G"], True, False, None),
        ("H+G", ["H", "G"], True, False, None),
        ("CE", ["CE"], True, False, None),
        ("H+CE", ["H", "CE"], True, False, None),
    ]

    rows: list[dict[str, Any]] = []
    rand_rows: list[dict[str, Any]] = []

    for name, feats, uses_y, uses_mem, _ in method_specs:
        if len(feats) == 1 and feats[0] in ("H", "G", "CE", "N_actual"):
            m = _eval_raw_feature(ext, feats[0])
        else:
            m = _fit_eval_failure_scorer(cal, ext, feats)
        rows.append(
            {
                "experiment": "exp1",
                "seed": seed,
                "method": name,
                "uses_y": uses_y,
                "uses_actual_memory": uses_mem,
                "auroc": m["auroc"],
                "auprc": m["auprc"],
                "conditional_log_loss": m["conditional_log_loss"],
                "n_external": m["n"],
                "random_memory_id": None,
            }
        )

    for r_id in range(cfg.n_random_memory):
        rand_seed = cfg.random_memory_base_seed + seed * 100 + r_id
        rand_lt = random_long_term_matched(actual_long_term, seed=rand_seed)
        cal_r = _score_split_from_checkpoint(params, ckpt["opt_state"], x_cal, y_cal, rand_lt)
        ext_r = _score_split_from_checkpoint(params, ckpt["opt_state"], x_ext, y_ext, rand_lt)
        cal_r["error"] = (~cal_r["correct"].astype(bool)).astype(int)
        ext_r["error"] = (~ext_r["correct"].astype(bool)).astype(int)
        cal_r["N_rand"] = cal_r["N_actual"]
        ext_r["N_rand"] = ext_r["N_actual"]

        for name, feats in [("N_rand", ["N_rand"]), ("H+N_rand", ["H", "N_rand"])]:
            if name == "N_rand":
                m = _eval_raw_feature(ext_r, "N_rand")
            else:
                m = _fit_eval_failure_scorer(cal_r, ext_r, feats)
            row = {
                "experiment": "exp1",
                "seed": seed,
                "method": name,
                "uses_y": True,
                "uses_actual_memory": False,
                "auroc": m["auroc"],
                "auprc": m["auprc"],
                "conditional_log_loss": m["conditional_log_loss"],
                "n_external": m["n"],
                "random_memory_id": r_id,
            }
            rows.append(row)
            rand_rows.append(row)

    return rows, rand_rows


def aggregate_experiment1(
    per_seed_rows: list[dict[str, Any]],
    rand_rows: list[dict[str, Any]],
    cfg: MemoryIdentificationConfig,
) -> pd.DataFrame:
    df = pd.DataFrame(per_seed_rows)
    # Pooled AUROC: concatenate per-seed calibrated scores is seed-dependent;
    # report per-seed mean/std and primary pooled recomputation from stored per-seed only for methods without random id.
    summary_rows: list[dict[str, Any]] = []
    primary = df[df["random_memory_id"].isna()]
    for method, sub in primary.groupby("method"):
        summary_rows.append(
            {
                "method": method,
                "uses_y": bool(sub["uses_y"].iloc[0]),
                "uses_actual_memory": bool(sub["uses_actual_memory"].iloc[0]),
                "auroc_mean": float(sub["auroc"].mean()),
                "auroc_std": float(sub["auroc"].std(ddof=0)),
                "auprc_mean": float(sub["auprc"].mean()),
                "conditional_log_loss_mean": float(sub["conditional_log_loss"].mean(skipna=True)),
                "n_seeds": int(len(sub)),
            }
        )

    rand_df = pd.DataFrame(rand_rows)
    if not rand_df.empty:
        for method in ("N_rand", "H+N_rand"):
            sub = rand_df[rand_df["method"] == method]
            summary_rows.append(
                {
                    "method": f"{method}_mean_over_random",
                    "uses_y": True,
                    "uses_actual_memory": False,
                    "auroc_mean": float(sub["auroc"].mean()),
                    "auroc_std": float(sub["auroc"].std(ddof=0)),
                    "auprc_mean": float(sub["auprc"].mean()),
                    "conditional_log_loss_mean": float(sub["conditional_log_loss"].mean(skipna=True)),
                    "n_seeds": int(sub["seed"].nunique()),
                }
            )

    summary = pd.DataFrame(summary_rows)

    # Paired deltas per seed
    delta_rows = []
    for seed in cfg.seeds:
        s = df[(df["seed"] == seed) & df["random_memory_id"].isna()]
        hnp = s[s["method"] == "H+N_actual"]["auroc"]
        hng = s[s["method"] == "H+G"]["auroc"]
        if len(hnp) and len(hng):
            delta_rows.append(
                {
                    "seed": seed,
                    "delta_gradient": float(hnp.iloc[0] - hng.iloc[0]),
                }
            )
        hnp_rand = rand_df[(rand_df["seed"] == seed) & (rand_df["method"] == "H+N_rand")]
        if len(hnp) and len(hnp_rand):
            delta_rows.append(
                {
                    "seed": seed,
                    "delta_memory_vs_random_mean": float(
                        hnp.iloc[0] - hnp_rand["auroc"].mean()
                    ),
                }
            )
    if delta_rows:
        ddf = pd.DataFrame(delta_rows)
        dm_mean, dm_lo, dm_hi = bootstrap_ci(
            ddf["delta_memory_vs_random_mean"].dropna().to_numpy(), cfg.ci_level
        )
        dg_mean, dg_lo, dg_hi = bootstrap_ci(
            ddf["delta_gradient"].dropna().to_numpy(), cfg.ci_level
        )
        summary.attrs["delta_memory_mean"] = dm_mean
        summary.attrs["delta_memory_ci"] = (dm_lo, dm_hi)
        summary.attrs["delta_gradient_mean"] = dg_mean
        summary.attrs["delta_gradient_ci"] = (dg_lo, dg_hi)

    return summary


def fixed_input_projection(
    X: np.ndarray,
    *,
    proj_dim: int,
    seed: int,
    block_size: int = 8192,
) -> np.ndarray:
    """Fixed Gaussian projection for h(x) -> R^{proj_dim} (blocked, no materialized P)."""
    return fixed_memory_projection(X, proj_dim=proj_dim, seed=seed, block_size=block_size)


def fit_h_plus_hproj_scorer(
    H_cal: np.ndarray,
    h_cal: np.ndarray,
    errors_cal: np.ndarray,
    *,
    proj_dim: int,
    seed: int,
) -> tuple[Pipeline, list[str], np.ndarray]:
    h_proj_cal = fixed_input_projection(h_cal, proj_dim=proj_dim, seed=seed)
    cal_df = pd.DataFrame({"H": H_cal, "error": errors_cal})
    for j in range(h_proj_cal.shape[1]):
        cal_df[f"h_{j}"] = h_proj_cal[:, j]
    cols = ["H", *[f"h_{j}" for j in range(h_proj_cal.shape[1])]]
    model = fit_failure_scorer(cal_df, cols, error_col="error")
    return model, cols, h_proj_cal


def run_experiment2_seed(cfg: MemoryIdentificationConfig, seed: int) -> list[dict[str, Any]]:
    mha_cfg = cfg.mha
    ckpt_path = _resolve_checkpoint(cfg, seed)
    ckpt = load_checkpoint(ckpt_path)
    params = ckpt["params"]
    M = load_memory_matrix_from_checkpoint(
        ckpt,
        source_dir=_memory_source_dir(cfg),
        task=cfg.task,
        seed=seed,
    )
    M_eff_actual = fixed_memory_projection(
        M,
        proj_dim=mha_cfg.memory_proj_dim,
        seed=seed + 7,
        block_size=mha_cfg.projection_block_size,
    )

    ds_cfg = ClinicalDatasetConfig(
        task=cfg.task,  # type: ignore[arg-type]
        data_dir=cfg.data_dir,  # type: ignore[arg-type]
        max_train=mha_cfg.max_train,
        max_cal=mha_cfg.max_cal,
        max_test=mha_cfg.max_test,
        max_external=mha_cfg.max_external,
    )
    bundle = load_clinical_bundle(ds_cfg)
    x_cal, y_cal, ids_cal = extract_split_arrays(bundle, "calibration")
    x_ext, y_ext, ids_ext = extract_split_arrays(bundle, "external")

    h_cal, _, H_cal, pred_cal = compute_representations(
        params, x_cal, num_classes=NUM_CLASSES_DERMA, batch_size=mha_cfg.batch_size
    )
    h_ext, _, H_ext, pred_ext = compute_representations(
        params, x_ext, num_classes=NUM_CLASSES_DERMA, batch_size=mha_cfg.batch_size
    )
    err_cal = (pred_cal != y_cal).astype(int)
    err_ext = (pred_ext != y_ext).astype(int)

    rows: list[dict[str, Any]] = []

    def record(method: str, label_free: bool, actual_mem: bool, m: dict[str, Any]) -> None:
        rows.append(
            {
                "experiment": "exp2",
                "seed": seed,
                "method": method,
                "label_free": label_free,
                "uses_actual_memory": actual_mem,
                "deferral_rate": m.get("deferral_rate"),
                "selective_risk": m.get("selective_risk"),
                "error_capture": m.get("error_capture"),
                "deferral_precision": m.get("deferral_precision"),
                "threshold": m.get("threshold"),
                "auroc": m.get("auroc"),
                "auprc": m.get("auprc"),
            }
        )

    # H-only baseline
    cal_h = pd.DataFrame({"H": H_cal, "error": err_cal})
    ext_h = pd.DataFrame({"H": H_ext, "error": err_ext})
    model_h = fit_failure_scorer(cal_h, ["H"], error_col="error")
    p_cal_h = deferral_probability(model_h, ["H"], cal_h)
    p_ext_h = deferral_probability(model_h, ["H"], ext_h)
    thr_h = calibrate_deferral_threshold(p_cal_h, target_deferral_rate=cfg.target_deferral_rate)
    m_h = deferral_metrics(err_ext, p_ext_h, thr_h)
    m_h.update(classification_metrics(err_ext, p_ext_h))
    record("H", True, False, m_h)

    # H + h(x) feature-only baseline (fixed projection to d_out for capacity match)
    model_hh, cols_hh, _ = fit_h_plus_hproj_scorer(
        H_cal, h_cal, err_cal, proj_dim=mha_cfg.d_out, seed=seed + 13
    )
    h_proj_ext = fixed_input_projection(h_ext, proj_dim=mha_cfg.d_out, seed=seed + 13)
    ext_hh = pd.DataFrame({"H": H_ext, "error": err_ext})
    for j in range(h_proj_ext.shape[1]):
        ext_hh[f"h_{j}"] = h_proj_ext[:, j]
    cal_hh = pd.DataFrame({"H": H_cal, "error": err_cal})
    h_proj_cal = fixed_input_projection(h_cal, proj_dim=mha_cfg.d_out, seed=seed + 13)
    for j in range(h_proj_cal.shape[1]):
        cal_hh[f"h_{j}"] = h_proj_cal[:, j]
    p_cal_hh = deferral_probability(model_hh, cols_hh, cal_hh)
    p_ext_hh = deferral_probability(model_hh, cols_hh, ext_hh)
    thr_hh = calibrate_deferral_threshold(p_cal_hh, target_deferral_rate=cfg.target_deferral_rate)
    m_hh = deferral_metrics(err_ext, p_ext_hh, thr_hh)
    m_hh.update(classification_metrics(err_ext, p_ext_hh))
    record("H+h", True, False, m_hh)

    # H + z_actual (MHA on actual memory)
    msa_actual, _ = train_msa_on_calibration(
        h_cal, H_cal, err_cal, jnp.asarray(M_eff_actual), mha_cfg, seed=seed
    )
    z_cal, _ = compute_z_and_attention(h_cal, jnp.asarray(M_eff_actual), msa_actual, mha_cfg)
    z_ext, _ = compute_z_and_attention(h_ext, jnp.asarray(M_eff_actual), msa_actual, mha_cfg)
    models_actual = fit_linear_scorers(H_cal, z_cal, err_cal)
    cal_z = pd.DataFrame({"H": H_cal, "error": err_cal})
    ext_z = pd.DataFrame({"H": H_ext, "error": err_ext})
    for j in range(z_ext.shape[1]):
        cal_z[f"z_{j}"] = z_cal[:, j]
        ext_z[f"z_{j}"] = z_ext[:, j]
    model_hn, cols_hn = models_actual["H+N"]
    p_cal_z = deferral_probability(model_hn, cols_hn, cal_z)
    p_ext_z = deferral_probability(model_hn, cols_hn, ext_z)
    thr_z = calibrate_deferral_threshold(p_cal_z, target_deferral_rate=cfg.target_deferral_rate)
    m_z = deferral_metrics(err_ext, p_ext_z, thr_z)
    m_z.update(classification_metrics(err_ext, p_ext_z))
    record("H+z_actual", True, True, m_z)

    # H + z_rand (MHA on random memory) — primary random realization per seed
    M_rand = random_memory_matrix_matched(M, seed=cfg.random_memory_base_seed + seed)
    M_eff_rand = fixed_memory_projection(
        M_rand,
        proj_dim=mha_cfg.memory_proj_dim,
        seed=seed + 7,
        block_size=mha_cfg.projection_block_size,
    )
    msa_rand, _ = train_msa_on_calibration(
        h_cal, H_cal, err_cal, jnp.asarray(M_eff_rand), mha_cfg, seed=seed + 10_000
    )
    z_cal_r, _ = compute_z_and_attention(h_cal, jnp.asarray(M_eff_rand), msa_rand, mha_cfg)
    z_ext_r, _ = compute_z_and_attention(h_ext, jnp.asarray(M_eff_rand), msa_rand, mha_cfg)
    models_rand = fit_linear_scorers(H_cal, z_cal_r, err_cal)
    cal_zr = pd.DataFrame({"H": H_cal, "error": err_cal})
    ext_zr = pd.DataFrame({"H": H_ext, "error": err_ext})
    for j in range(z_ext_r.shape[1]):
        cal_zr[f"z_{j}"] = z_cal_r[:, j]
        ext_zr[f"z_{j}"] = z_ext_r[:, j]
    model_hnr, cols_hnr = models_rand["H+N"]
    p_cal_zr = deferral_probability(model_hnr, cols_hnr, cal_zr)
    p_ext_zr = deferral_probability(model_hnr, cols_hnr, ext_zr)
    thr_zr = calibrate_deferral_threshold(p_cal_zr, target_deferral_rate=cfg.target_deferral_rate)
    m_zr = deferral_metrics(err_ext, p_ext_zr, thr_zr)
    m_zr.update(classification_metrics(err_ext, p_ext_zr))
    record("H+z_rand", True, False, m_zr)

    use_assoc = cfg.use_associative_memory or resolve_associative_mode(mha_cfg, seed)
    if use_assoc and associative_artifacts_exist(_memory_source_dir(cfg), cfg.task, seed):
        _state, traj_bundle, assoc_cfg = load_associative_artifacts(
            _memory_source_dir(cfg),
            cfg.task,
            seed,
        )
        z_cal_mem, _ = batch_associative_z_memory(
            h_cal,
            traj_bundle,
            assoc_cfg,
            d_out=mha_cfg.d_out,
        )
        z_ext_mem, _ = batch_associative_z_memory(
            h_ext,
            traj_bundle,
            assoc_cfg,
            d_out=mha_cfg.d_out,
        )
        cal_hhz = pd.DataFrame({"H": H_cal, "error": err_cal})
        ext_hhz = pd.DataFrame({"H": H_ext, "error": err_ext})
        for j in range(z_ext_mem.shape[1]):
            cal_hhz[f"z_{j}"] = z_cal_mem[:, j]
            ext_hhz[f"z_{j}"] = z_ext_mem[:, j]
        h_proj_cal = fixed_input_projection(h_cal, proj_dim=mha_cfg.d_out, seed=seed + 13)
        h_proj_ext = fixed_input_projection(h_ext, proj_dim=mha_cfg.d_out, seed=seed + 13)
        for j in range(h_proj_ext.shape[1]):
            cal_hhz[f"h_{j}"] = h_proj_cal[:, j]
            ext_hhz[f"h_{j}"] = h_proj_ext[:, j]
        z_cols = [f"z_{j}" for j in range(z_ext_mem.shape[1])]
        h_cols = [f"h_{j}" for j in range(h_proj_ext.shape[1])]
        model_hhz = fit_failure_scorer(cal_hhz, ["H", *h_cols, *z_cols], error_col="error")
        p_cal_hhz = deferral_probability(model_hhz, ["H", *h_cols, *z_cols], cal_hhz)
        p_ext_hhz = deferral_probability(model_hhz, ["H", *h_cols, *z_cols], ext_hhz)
        thr_hhz = calibrate_deferral_threshold(p_cal_hhz, target_deferral_rate=cfg.target_deferral_rate)
        m_hhz = deferral_metrics(err_ext, p_ext_hhz, thr_hhz)
        m_hhz.update(classification_metrics(err_ext, p_ext_hhz))
        record("H+h+z_memory", True, True, m_hhz)

        z_cal_rand = _z_memory_from_randomized_trajectories(
            h_cal,
            traj_bundle,
            seed=cfg.random_memory_base_seed + seed,
            assoc_cfg=assoc_cfg,
            d_out=mha_cfg.d_out,
        )
        z_ext_rand = _z_memory_from_randomized_trajectories(
            h_ext,
            traj_bundle,
            seed=cfg.random_memory_base_seed + seed,
            assoc_cfg=assoc_cfg,
            d_out=mha_cfg.d_out,
        )
        cal_hhzr = pd.DataFrame({"H": H_cal, "error": err_cal})
        ext_hhzr = pd.DataFrame({"H": H_ext, "error": err_ext})
        for j in range(z_ext_rand.shape[1]):
            cal_hhzr[f"z_{j}"] = z_cal_rand[:, j]
            ext_hhzr[f"z_{j}"] = z_ext_rand[:, j]
        for j in range(h_proj_ext.shape[1]):
            cal_hhzr[f"h_{j}"] = h_proj_cal[:, j]
            ext_hhzr[f"h_{j}"] = h_proj_ext[:, j]
        model_hhzr = fit_failure_scorer(cal_hhzr, ["H", *h_cols, *z_cols], error_col="error")
        p_cal_hhzr = deferral_probability(model_hhzr, ["H", *h_cols, *z_cols], cal_hhzr)
        p_ext_hhzr = deferral_probability(model_hhzr, ["H", *h_cols, *z_cols], ext_hhzr)
        thr_hhzr = calibrate_deferral_threshold(p_cal_hhzr, target_deferral_rate=cfg.target_deferral_rate)
        m_hhzr = deferral_metrics(err_ext, p_ext_hhzr, thr_hhzr)
        m_hhzr.update(classification_metrics(err_ext, p_ext_hhzr))
        record("H+h+z_rand", True, False, m_hhzr)

        _ = randomize_associative_memory(_state, seed=cfg.random_memory_base_seed + seed)

    return rows


def classify_evidence(exp1_summary: pd.DataFrame, exp2_df: pd.DataFrame) -> Literal["A", "B", "C"]:
    def _get(summary: pd.DataFrame, method: str, col: str = "auroc_mean") -> float:
        if summary.empty or "method" not in summary.columns or col not in summary.columns:
            return float("nan")
        row = summary[summary["method"] == method]
        return float(row[col].iloc[0]) if len(row) else float("nan")

    hnp = _get(exp1_summary, "H+N_actual")
    hng = _get(exp1_summary, "H+G")
    hnp_rand = _get(exp1_summary, "H+N_rand_mean_over_random")

    mem_beats_rand = np.isfinite(hnp) and np.isfinite(hnp_rand) and (hnp - hnp_rand) > 0.01
    mem_beats_grad = np.isfinite(hnp) and np.isfinite(hng) and (hnp - hng) > 0.005

    if not exp2_df.empty:
        pooled = exp2_df.groupby("method").agg(
            selective_risk=("selective_risk", "mean"),
            deferral_precision=("deferral_precision", "mean"),
        )
        z_act = pooled.loc["H+z_actual", "selective_risk"] if "H+z_actual" in pooled.index else float("nan")
        z_rand = pooled.loc["H+z_rand", "selective_risk"] if "H+z_rand" in pooled.index else float("nan")
        h_h = pooled.loc["H+h", "selective_risk"] if "H+h" in pooled.index else float("nan")
        mha_beats_rand = np.isfinite(z_act) and np.isfinite(z_rand) and z_act < z_rand
        mha_beats_h = np.isfinite(z_act) and np.isfinite(h_h) and z_act < h_h
    else:
        mha_beats_rand = False
        mha_beats_h = False

    if mem_beats_rand and mem_beats_grad and mha_beats_rand and mha_beats_h:
        return "A"
    if mem_beats_rand or mha_beats_rand:
        return "B"
    return "C"


def write_interpretation(
    path: Path,
    *,
    exp1_per_seed: pd.DataFrame,
    exp1_summary: pd.DataFrame,
    exp1_rand: pd.DataFrame,
    exp2_per_seed: pd.DataFrame,
    evidence_class: str,
) -> None:
    lines = [
        "# Memory Identification — Interpretation",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## What was tested",
        "",
        "- **Experiment 1:** Whether label-dependent novelty from actual optimization-history memory "
        "beats gradient magnitude, supervised loss, and matched random-memory controls on DermaMNIST external shift.",
        "- **Experiment 2:** Whether label-free MHA retrieval from actual memory beats penultimate-feature-only "
        "and random-memory MHA baselines under simulated human deferral.",
        "",
        "## Experiment 1 summary (mean AUROC over seeds)",
        "",
        exp1_summary.to_string(index=False),
        "",
        "## Key paired comparisons (Experiment 1)",
        "",
    ]

    def _mean_delta(method_a: str, method_b: str) -> float:
        a = exp1_per_seed[exp1_per_seed["method"] == method_a].set_index("seed")["auroc"]
        b = exp1_per_seed[exp1_per_seed["method"] == method_b].set_index("seed")["auroc"]
        common = a.index.intersection(b.index)
        if len(common) == 0:
            return float("nan")
        return float((a.loc[common] - b.loc[common]).mean())

    d_mem = _mean_delta("H+N_actual", "H+N_rand")
    if not exp1_rand.empty:
        d_mem = float(
            exp1_per_seed[exp1_per_seed["method"] == "H+N_actual"]["auroc"].mean()
            - exp1_rand[exp1_rand["method"] == "H+N_rand"]["auroc"].mean()
        )
    d_grad = _mean_delta("H+N_actual", "H+G")

    lines += [
        f"- Δ_memory = AUROC(H+N_actual) − mean AUROC(H+N_rand): **{d_mem:+.4f}**",
        f"- Δ_gradient = AUROC(H+N_actual) − AUROC(H+G): **{d_grad:+.4f}**",
        "",
        "## Experiment 2 summary (mean selective risk over seeds; lower is better)",
        "",
    ]
    if not exp2_per_seed.empty:
        lines.append(
            exp2_per_seed.groupby("method")[["selective_risk", "deferral_precision", "error_capture", "deferral_rate"]]
            .mean()
            .reset_index()
            .to_string(index=False)
        )
    else:
        lines.append("(not run)")

    lines += [
        "",
        "## Did actual memory beat random memory?",
        "",
        f"- Experiment 1 (H+N): {'yes' if d_mem > 0.01 else 'weak/no'} (Δ={d_mem:+.4f})",
        "",
        "## Did actual memory beat gradient/feature controls?",
        "",
        f"- vs H+G: {'yes' if d_grad > 0.005 else 'weak/no'} (Δ={d_grad:+.4f})",
        "",
        "## Evidence classification",
        "",
        f"**{evidence_class}** — "
        + {
            "A": "Strong mechanism support: actual memory beats matched random and gradient/feature controls.",
            "B": "Partial support: memory helps but simpler controls explain part of the gain.",
            "C": "Weak support: random memory or feature baselines are competitive; mechanism claim should be weakened.",
        }.get(evidence_class, "unknown"),
        "",
        "## Notes",
        "",
        "- These are identification experiments, not score-optimization runs.",
        "- MedMNIST benchmark; simulated deferral only.",
        "- Do not interpret CE/label-dependent scores as deployable routing methods.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_cached_experiment1_partial(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed_path = out / "experiment1_per_seed.csv"
    rand_path = out / "experiment1_random_memory_distribution.csv"
    per_seed = pd.read_csv(per_seed_path) if per_seed_path.exists() else pd.DataFrame()
    rand_df = pd.read_csv(rand_path) if rand_path.exists() else pd.DataFrame()
    return per_seed, rand_df


def _completed_seeds(df: pd.DataFrame) -> set[int]:
    if df.empty or "seed" not in df.columns:
        return set()
    return {int(s) for s in df["seed"].dropna().unique()}


def _load_cached_experiment2_partial(out: Path) -> pd.DataFrame:
    exp2_path = out / "experiment2_mha_memory_controls.csv"
    if not exp2_path.exists():
        return pd.DataFrame()
    return pd.read_csv(exp2_path)


def run_memory_identification(cfg: MemoryIdentificationConfig | None = None) -> dict[str, Any]:
    cfg = (cfg or MemoryIdentificationConfig()).resolve()

    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)

    exp1_rows: list[dict[str, Any]] = []
    exp1_rand: list[dict[str, Any]] = []
    exp2_rows: list[dict[str, Any]] = []

    if cfg.run_experiment1:
        exp1_cached, exp1_rand_cached = _load_cached_experiment1_partial(out)
        done_exp1 = _completed_seeds(exp1_cached)
        exp1_rows.extend(exp1_cached.to_dict("records"))
        exp1_rand.extend(exp1_rand_cached.to_dict("records"))
        if done_exp1:
            print(f"[exp1] loaded cached seeds {sorted(done_exp1)}", flush=True)
        for seed in cfg.seeds:
            if int(seed) in done_exp1:
                continue
            print(f"[exp1] seed {seed}", flush=True)
            rows, rand_rows = run_experiment1_seed(cfg, seed)
            exp1_rows.extend(rows)
            exp1_rand.extend(rand_rows)
            pd.DataFrame(exp1_rows).to_csv(out / "experiment1_per_seed.csv", index=False)
            pd.DataFrame(exp1_rand).to_csv(out / "experiment1_random_memory_distribution.csv", index=False)
    else:
        exp1_cached, exp1_rand_cached = _load_cached_experiment1_partial(out)
        if not exp1_cached.empty:
            exp1_rows = exp1_cached.to_dict("records")
            exp1_rand = exp1_rand_cached.to_dict("records")
            print(f"[exp1] loaded cached results ({len(exp1_cached)} rows)", flush=True)

    exp1_per_seed = pd.DataFrame(exp1_rows)
    exp1_rand_df = pd.DataFrame(exp1_rand)
    exp1_summary = aggregate_experiment1(exp1_rows, exp1_rand, cfg) if exp1_rows else pd.DataFrame()

    if cfg.run_experiment2:
        exp2_cached = _load_cached_experiment2_partial(out)
        done_exp2 = _completed_seeds(exp2_cached)
        exp2_rows.extend(exp2_cached.to_dict("records"))
        if done_exp2:
            print(f"[exp2] loaded cached seeds {sorted(done_exp2)}", flush=True)
        for seed in cfg.seeds:
            if int(seed) in done_exp2:
                continue
            print(f"[exp2] seed {seed}", flush=True)
            seed_rows = run_experiment2_seed(cfg, seed)
            exp2_rows.extend(seed_rows)
            pd.DataFrame(exp2_rows).to_csv(out / "experiment2_mha_memory_controls.csv", index=False)

    exp2_per_seed = pd.DataFrame(exp2_rows)
    evidence = classify_evidence(exp1_summary, exp2_per_seed)

    config_path = out / "config.json"
    config_payload = {
        "task": cfg.task,
        "seeds": list(cfg.seeds),
        "n_random_memory": cfg.n_random_memory,
        "random_memory_base_seed": cfg.random_memory_base_seed,
        "source_experiment_dir": str(cfg.source_experiment_dir),
        "output_dir": str(cfg.output_dir),
        "target_deferral_rate": cfg.target_deferral_rate,
        "run_experiment1": cfg.run_experiment1,
        "run_experiment2": cfg.run_experiment2,
    }
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    exp1_path = out / "experiment1_gradient_memory_controls.csv"
    exp1_per_seed_path = out / "experiment1_per_seed.csv"
    exp1_rand_path = out / "experiment1_random_memory_distribution.csv"
    exp1_summary_path = out / "experiment1_aggregate.csv"
    exp2_path = out / "experiment2_mha_memory_controls.csv"

    exp1_per_seed.to_csv(exp1_path, index=False)
    exp1_per_seed.to_csv(exp1_per_seed_path, index=False)
    exp1_rand_df.to_csv(exp1_rand_path, index=False)
    exp1_summary.to_csv(exp1_summary_path, index=False)
    exp2_per_seed.to_csv(exp2_path, index=False)

    write_interpretation(
        out / "interpretation.md",
        exp1_per_seed=exp1_per_seed,
        exp1_summary=exp1_summary,
        exp1_rand=exp1_rand_df,
        exp2_per_seed=exp2_per_seed,
        evidence_class=evidence,
    )

    return {
        "config": cfg,
        "experiment1": exp1_per_seed,
        "experiment1_summary": exp1_summary,
        "experiment1_random": exp1_rand_df,
        "experiment2": exp2_per_seed,
        "evidence_class": evidence,
        "output_dir": out,
    }


def main() -> None:
    cfg = MemoryIdentificationConfig()
    if "--no-exp2" in sys.argv:
        cfg.run_experiment2 = False
    if "--no-exp1" in sys.argv:
        cfg.run_experiment1 = False
    if "--n-rand" in sys.argv:
        idx = sys.argv.index("--n-rand")
        cfg.n_random_memory = int(sys.argv[idx + 1])
    if "--train-missing" in sys.argv:
        cfg.train_if_checkpoint_missing = True
    results = run_memory_identification(cfg.resolve())
    print(f"Evidence class: {results['evidence_class']}")
    print(f"Artifacts: {results['output_dir']}")


if __name__ == "__main__":
    main()
