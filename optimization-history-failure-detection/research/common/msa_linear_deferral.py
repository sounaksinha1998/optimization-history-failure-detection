"""Final MSA + linear deferral experiment: h(x) attends to frozen V5B memory."""

from __future__ import annotations

import json
import pickle
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from optimizer.associative_memory import AssociativeMemoryConfig
from research.common.clinical_datasets import ClinicalDatasetConfig, load_clinical_bundle
from research.common.clinical_failure import aurc_from_curve, risk_coverage_curve
from research.common.clinical_training import checkpoint_path, load_checkpoint
from research.common.error_prediction import predict_error_probability
from research.common.memory import (
    associative_artifacts_exist,
    extract_nrm_v2_state,
    load_associative_artifacts,
    memory_state_summary,
)
from research.common.msa import batch_associative_z_memory
from research.common.resnet import resnet18_apply, resnet18_features, resnet18_probs
from research.common.trajectory_store import MemoryCheckpointBundle

ScorerName = Literal["H", "N", "H+N", "H+N_perm"]
PRIMARY_DEFERRAL_RATE = 0.20


@dataclass
class MSALinearDeferralConfig:
    repo_root: Path | None = None
    source_experiment_dir: Path | None = None
    data_dir: Path | None = None
    output_dir: Path | None = None
    seeds: tuple[int, ...] = (42, 123, 456)
    task: str = "dermamnist"
    num_classes: int = 7
    n_heads: int = 4
    d_k: int = 64
    d_out: int = 128
    memory_proj_dim: int = 512
    projection_block_size: int = 8192
    msa_train_steps: int = 400
    msa_learning_rate: float = 1e-2
    msa_l2: float = 1e-4
    target_deferral_rate: float = PRIMARY_DEFERRAL_RATE
    n_h_bins_perm: int = 20
    permutation_seed: int = 42
    n_bootstrap: int = 2000
    bootstrap_seed: int = 20260831
    ci_level: float = 0.95
    batch_size: int = 64
    show_plots: bool = True
    save_artifacts: bool = True
    max_train: int | None = 2000
    max_cal: int | None = 500
    max_test: int | None = 1000
    max_external: int | None = 1000
    use_associative_memory: bool = False
    associative_memory: AssociativeMemoryConfig = field(default_factory=AssociativeMemoryConfig)

    def resolve_paths(self, cwd: Path | None = None) -> MSALinearDeferralConfig:
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
            self.output_dir = root / "research" / "dermamnist_msa_linear_deferral"
        return self


def flatten_memory_matrix(mem_state: Any) -> np.ndarray:
    """M ∈ R^{T×D} from frozen long-term V5B levels."""
    rows = []
    for level in mem_state.long_term:
        leaves = jax.tree_util.tree_leaves(level)
        rows.append(np.concatenate([np.asarray(x, dtype=np.float32).reshape(-1) for x in leaves]))
    return np.stack(rows, axis=0)


def load_memory_matrix_from_checkpoint(
    ckpt: dict[str, Any],
    *,
    source_dir: Path | None,
    task: str,
    seed: int,
) -> np.ndarray:
    """Load M ∈ R^{T×D} from companion .npz when available, else flatten checkpoint state."""
    if source_dir is not None:
        npz_path = source_dir / "memory" / f"{task}_seed{seed}_memory.npz"
        if npz_path.exists():
            data = np.load(npz_path, allow_pickle=True)
            rows = [np.asarray(row, dtype=np.float32).reshape(-1) for row in data["long_term"]]
            return np.stack(rows, axis=0)
    mem_state = extract_nrm_v2_state(ckpt["opt_state"])
    return flatten_memory_matrix(mem_state)


def resolve_associative_mode(cfg: MSALinearDeferralConfig, seed: int) -> bool:
    """Return whether to use label-free associative trajectories for this seed."""
    if cfg.use_associative_memory:
        return True
    if cfg.source_experiment_dir is None:
        return False
    return associative_artifacts_exist(cfg.source_experiment_dir, cfg.task, seed)


def load_associative_trajectory_from_checkpoint(
    *,
    source_dir: Path,
    task: str,
    seed: int,
) -> tuple[MemoryCheckpointBundle, AssociativeMemoryConfig]:
    """Load shared-grid memory checkpoints and associative config."""
    _state, bundle, assoc_cfg = load_associative_artifacts(source_dir, task, seed)
    return bundle, assoc_cfg


def compute_z_from_associative_trajectories(
    h: np.ndarray,
    checkpoints: MemoryCheckpointBundle,
    cfg: MSALinearDeferralConfig,
    *,
    assoc_cfg: AssociativeMemoryConfig | None = None,
    attention_params: dict[str, jnp.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Label-free z_memory from frozen h_T(x) replayed against {M_t^j}."""
    resolved_cfg = assoc_cfg or cfg.associative_memory
    return batch_associative_z_memory(
        h,
        checkpoints,
        resolved_cfg,
        attention_params=attention_params,
        d_out=cfg.d_out,
    )


def seed_from_ids(sample_ids: np.ndarray) -> int:
    ids = np.asarray(sample_ids).reshape(-1)
    if ids.size == 0:
        return 0
    return hash(str(ids[0])) % 10_007


def fixed_memory_projection(
    M: np.ndarray,
    *,
    proj_dim: int,
    seed: int,
    block_size: int = 8192,
) -> np.ndarray:
    """M_eff = M @ P with fixed Gaussian P, without materializing P (blocked matmul).

    Equivalent to P ~ N(0, 1/proj_dim)^{D×proj_dim} but uses O(T·block_size) peak memory
    instead of O(D·proj_dim).
    """
    M = np.asarray(M, dtype=np.float32)
    t, d = M.shape
    m_eff = np.zeros((t, proj_dim), dtype=np.float32)
    scale = np.float32(1.0 / np.sqrt(proj_dim))
    for j in range(proj_dim):
        col_rng = np.random.default_rng(int(seed) + j * 104_729)
        acc = np.zeros(t, dtype=np.float64)
        for start in range(0, d, block_size):
            end = min(d, start + block_size)
            w = col_rng.standard_normal(end - start).astype(np.float32)
            acc += M[:, start:end] @ w
        m_eff[:, j] = (acc * scale).astype(np.float32)
    return m_eff


def normalized_entropy_from_probs(probs: np.ndarray, num_classes: int, eps: float = 1e-12) -> np.ndarray:
    safe = np.clip(probs, eps, 1.0)
    h = -np.sum(safe * np.log(safe), axis=-1)
    return (h / np.log(num_classes)).astype(np.float32)


def extract_split_arrays(
    bundle: Any,
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mapping = {
        "calibration": ("x_cal", "y_cal", "cal"),
        "external": ("x_external", "y_external", "external"),
        "test": ("x_test", "y_test", "test"),
    }
    if split not in mapping:
        raise ValueError(split)
    x_attr, y_attr, id_key = mapping[split]
    return bundle.__dict__[x_attr], bundle.__dict__[y_attr], bundle.sample_ids[id_key]


def compute_representations(
    params: dict,
    x: np.ndarray,
    *,
    num_classes: int,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return h(x), probs, H(x), predictions."""
    h_list: list[np.ndarray] = []
    prob_list: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        xb = jnp.asarray(x[start : start + batch_size], dtype=jnp.float32)
        h_list.append(np.asarray(resnet18_features(params, xb), dtype=np.float32))
        prob_list.append(np.asarray(resnet18_probs(params, xb), dtype=np.float32))
    h = np.concatenate(h_list, axis=0)
    probs = np.concatenate(prob_list, axis=0)
    H = normalized_entropy_from_probs(probs, num_classes)
    preds = probs.argmax(axis=-1)
    return h, probs, H, preds


def init_msa_params(
    key: jax.Array,
    *,
    h_dim: int,
    mem_dim: int,
    n_heads: int,
    d_k: int,
    d_out: int,
) -> dict[str, jnp.ndarray]:
    keys = jax.random.split(key, n_heads * 3 + 1)
    params: dict[str, jnp.ndarray] = {"W_O": jax.random.normal(keys[0], (n_heads * d_k, d_out)) * 0.01}
    idx = 1
    for k in range(n_heads):
        params[f"W_Q_{k}"] = jax.random.normal(keys[idx], (h_dim, d_k)) * 0.01
        params[f"W_K_{k}"] = jax.random.normal(keys[idx + 1], (mem_dim, d_k)) * 0.01
        params[f"W_V_{k}"] = jax.random.normal(keys[idx + 2], (mem_dim, d_k)) * 0.01
        idx += 3
    return params


def msa_forward_batch(
    h_batch: jnp.ndarray,
    M_eff: jnp.ndarray,
    params: dict[str, jnp.ndarray],
    *,
    n_heads: int,
    d_k: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Batched multi-head attention. h_batch: (B, h_dim) -> z: (B, d_out), attn: (B, n_heads, T)."""
    if h_batch.ndim == 1:
        h_batch = h_batch[None, :]
    z_heads: list[jnp.ndarray] = []
    attn_heads: list[jnp.ndarray] = []
    scale = jnp.sqrt(jnp.asarray(d_k, dtype=jnp.float32))
    for k in range(n_heads):
        q = h_batch @ params[f"W_Q_{k}"]
        k_mat = M_eff @ params[f"W_K_{k}"]
        v_mat = M_eff @ params[f"W_V_{k}"]
        logits = (q @ k_mat.T) / scale
        a = jax.nn.softmax(logits, axis=-1)
        z_heads.append(a @ v_mat)
        attn_heads.append(a)
    z_cat = jnp.concatenate(z_heads, axis=-1)
    z_out = z_cat @ params["W_O"]
    attn = jnp.stack(attn_heads, axis=1)
    return z_out, attn


def msa_forward(
    h: jnp.ndarray,
    M_eff: jnp.ndarray,
    params: dict[str, jnp.ndarray],
    *,
    n_heads: int,
    d_k: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Single-sample forward; returns z (d_out,) and attn (n_heads, T)."""
    z_b, attn_b = msa_forward_batch(h, M_eff, params, n_heads=n_heads, d_k=d_k)
    return z_b.reshape(-1), attn_b.reshape(attn_b.shape[1], attn_b.shape[2])


def train_msa_on_calibration(
    h_cal: np.ndarray,
    H_cal: np.ndarray,
    y_err_cal: np.ndarray,
    M_eff: jnp.ndarray,
    cfg: MSALinearDeferralConfig,
    *,
    seed: int,
) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
    """Train MSA + provisional linear head on calibration only (BCE)."""
    h_dim = h_cal.shape[1]
    mem_dim = M_eff.shape[1]
    key = jax.random.PRNGKey(seed + 17_000)
    msa_params = init_msa_params(
        key,
        h_dim=h_dim,
        mem_dim=mem_dim,
        n_heads=cfg.n_heads,
        d_k=cfg.d_k,
        d_out=cfg.d_out,
    )
    head_params = {
        "b": jnp.array(0.0),
        "w_H": jnp.array(0.0),
        "w_N": jax.random.normal(jax.random.PRNGKey(seed + 99), (cfg.d_out,)) * 0.01,
    }
    all_params = {"msa": msa_params, "head": head_params}

    def loss_fn(params, h_batch, H_batch, y_batch):
        z, _ = msa_forward_batch(
            h_batch, M_eff, params["msa"], n_heads=cfg.n_heads, d_k=cfg.d_k
        )
        logit = params["head"]["b"] + params["head"]["w_H"] * H_batch + z @ params["head"]["w_N"]
        p = jax.nn.sigmoid(logit)
        eps = 1e-6
        bce = -(y_batch * jnp.log(p + eps) + (1 - y_batch) * jnp.log(1 - p + eps)).mean()
        reg = cfg.msa_l2 * sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(params))
        return bce + reg

    optimizer = optax.adam(cfg.msa_learning_rate)

    @jax.jit
    def train_step(params, opt_state, h_batch, H_batch, y_batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, h_batch, H_batch, y_batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    opt_state = optimizer.init(all_params)
    h_j = jnp.asarray(h_cal)
    H_j = jnp.asarray(H_cal)
    y_j = jnp.asarray(y_err_cal.astype(np.float32))
    n = len(h_cal)
    for step in range(cfg.msa_train_steps):
        idx = np.random.default_rng(seed + step).integers(0, n, size=min(cfg.batch_size, n))
        all_params, opt_state, loss = train_step(
            all_params, opt_state, h_j[idx], H_j[idx], y_j[idx]
        )
        if step % 100 == 0:
            pass
    return all_params["msa"], all_params["head"]


def compute_z_and_attention(
    h: np.ndarray,
    M_eff: jnp.ndarray,
    msa_params: dict[str, jnp.ndarray],
    cfg: MSALinearDeferralConfig,
    *,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    h_j = jnp.asarray(h, dtype=jnp.float32)
    z_parts: list[np.ndarray] = []
    attn_parts: list[np.ndarray] = []
    for start in range(0, len(h), batch_size):
        zb, ab = msa_forward_batch(
            h_j[start : start + batch_size],
            M_eff,
            msa_params,
            n_heads=cfg.n_heads,
            d_k=cfg.d_k,
        )
        z_parts.append(np.asarray(zb))
        attn_parts.append(np.asarray(ab))
    return np.concatenate(z_parts, axis=0), np.concatenate(attn_parts, axis=0)


def fit_linear_scorers(
    H: np.ndarray,
    z: np.ndarray,
    errors: np.ndarray,
) -> dict[str, tuple[Pipeline, list[str]]]:
    cal_df = pd.DataFrame({"H": H, "error": errors})
    for j in range(z.shape[1]):
        cal_df[f"z_{j}"] = z[:, j]
    z_cols = [f"z_{j}" for j in range(z.shape[1])]
    models: dict[str, tuple[Pipeline, list[str]]] = {}
    specs = {
        "H": ["H"],
        "N": z_cols,
        "H+N": ["H", *z_cols],
    }
    for name, cols in specs.items():
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=0)),
            ]
        )
        model.fit(cal_df[cols], cal_df["error"])
        models[name] = (model, cols)
    return models


def deferral_probability(model: Pipeline, cols: list[str], df: pd.DataFrame) -> np.ndarray:
    return predict_error_probability(model, df, cols)


def calibrate_deferral_threshold(probs: np.ndarray, *, target_deferral_rate: float) -> float:
    valid = probs[np.isfinite(probs)]
    if len(valid) == 0:
        return float("nan")
    sorted_desc = np.sort(valid)[::-1]
    k = max(1, int(round(target_deferral_rate * len(sorted_desc))))
    return float(sorted_desc[min(k - 1, len(sorted_desc) - 1)])


def deferral_metrics(
    errors: np.ndarray,
    defer_probs: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    errors = np.asarray(errors, dtype=int)
    probs = np.asarray(defer_probs, dtype=float)
    mask = np.isfinite(probs)
    errors = errors[mask]
    probs = probs[mask]
    n = len(errors)
    if n == 0:
        return {}
    deferred = probs >= threshold
    accepted = ~deferred
    n_deferred = int(deferred.sum())
    n_accepted = int(accepted.sum())
    n_errors = int(errors.sum())
    n_errors_deferred = int(errors[deferred].sum()) if n_deferred else 0
    r_all = float(errors.mean())
    r_sel = float(errors[accepted].mean()) if n_accepted else float("nan")
    return {
        "n": n,
        "deferral_rate": float(n_deferred / n),
        "coverage": float(n_accepted / n),
        "selective_risk": r_sel,
        "selective_accuracy": float(1.0 - r_sel) if n_accepted else float("nan"),
        "error_rate_deferred": float(errors[deferred].mean()) if n_deferred else float("nan"),
        "deferral_precision": float(n_errors_deferred / n_deferred) if n_deferred else float("nan"),
        "error_capture": float(n_errors_deferred / n_errors) if n_errors else float("nan"),
        "n_deferred": n_deferred,
        "n_errors": n_errors,
        "n_errors_deferred": n_errors_deferred,
        "risk_all": r_all,
        "risk_reduction": float(r_all - r_sel) if n_accepted else float("nan"),
        "threshold": float(threshold),
    }


def classification_metrics(errors: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(scores)
    y = errors[mask]
    s = scores[mask]
    if len(np.unique(y)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "aurc": float("nan")}
    curve = risk_coverage_curve(y, s)
    return {
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
        "aurc": float(aurc_from_curve(curve)),
    }


def permute_z_within_h_bins(H: np.ndarray, z: np.ndarray, *, n_bins: int, seed: int) -> np.ndarray:
    out = z.copy()
    edges = np.quantile(H, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    bin_ids = np.digitize(H, edges[1:-1], right=True)
    rng = np.random.default_rng(seed)
    for b in np.unique(bin_ids):
        idx = np.where(bin_ids == b)[0]
        if len(idx) <= 1:
            continue
        perm = idx[rng.permutation(len(idx))]
        out[idx] = z[perm]
    return out


def bootstrap_pooled_deferred(
    ps: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    errors = ps["error"].to_numpy(dtype=int)
    deferred_h = ps["deferred_H"].to_numpy(dtype=bool)
    deferred_hn = ps["deferred_HN"].to_numpy(dtype=bool)
    probs_h = ps["defer_prob_H"].to_numpy(dtype=float)
    probs_hn = ps["defer_prob_HN"].to_numpy(dtype=float)
    n = len(errors)
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        e = errors[idx]
        m_h = metrics_from_deferred_columns(e, probs_h[idx], deferred_h[idx])
        m_hn = metrics_from_deferred_columns(e, probs_hn[idx], deferred_hn[idx])
        rows.append(
            {
                "bootstrap_id": b,
                "seed": "pooled",
                "delta_selective_risk": m_h["selective_risk"] - m_hn["selective_risk"],
                "delta_deferral_precision": m_hn["deferral_precision"] - m_h["deferral_precision"],
                "delta_error_capture": m_hn["error_capture"] - m_h["error_capture"],
                "delta_auroc": float(roc_auc_score(e, probs_hn[idx]) - roc_auc_score(e, probs_h[idx]))
                if len(np.unique(e)) > 1
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_paired(
    errors: np.ndarray,
    probs_h: np.ndarray,
    probs_hn: np.ndarray,
    thr_h: float,
    thr_hn: float,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    n = len(errors)
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        m_h = deferral_metrics(errors[idx], probs_h[idx], thr_h)
        m_hn = deferral_metrics(errors[idx], probs_hn[idx], thr_hn)
        rows.append(
            {
                "bootstrap_id": b,
                "delta_selective_risk": m_h["selective_risk"] - m_hn["selective_risk"],
                "delta_deferral_precision": m_hn["deferral_precision"] - m_h["deferral_precision"],
                "delta_error_capture": m_hn["error_capture"] - m_h["error_capture"],
                "delta_auroc": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, ci_level: float) -> tuple[float, float, float]:
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    a = (1 - ci_level) / 2
    return float(np.mean(v)), float(np.percentile(v, 100 * a)), float(np.percentile(v, 100 * (1 - a)))


def preregistered_verdict(
    primary_h: dict[str, Any],
    primary_hn: dict[str, Any],
    boot_df: pd.DataFrame,
    perm_hn: dict[str, Any],
    perm_null: dict[str, Any],
    per_seed_pass: list[bool],
    *,
    ci_level: float,
) -> dict[str, Any]:
    dr_mean, dr_lo, dr_hi = bootstrap_ci(boot_df["delta_selective_risk"].to_numpy(), ci_level)
    dp_mean, dp_lo, dp_hi = bootstrap_ci(boot_df["delta_deferral_precision"].to_numpy(), ci_level)
    c1 = primary_hn["selective_risk"] < primary_h["selective_risk"]
    c2 = bool(dr_lo > 0)
    c3 = bool(all(per_seed_pass) or sum(per_seed_pass) >= 2)
    c4 = perm_hn["selective_risk"] < perm_null["selective_risk"]
    passed = bool(c1 and c2 and c3 and c4)
    return {
        "criterion1_risk_HN_lower_than_H": c1,
        "criterion2_delta_risk_ci_excludes_zero": c2,
        "criterion3_not_contradicted_across_seeds": c3,
        "criterion4_beats_permutation_null": c4,
        "delta_selective_risk_mean": dr_mean,
        "delta_selective_risk_ci_low": dr_lo,
        "delta_selective_risk_ci_high": dr_hi,
        "delta_deferral_precision_mean": dp_mean,
        "delta_deferral_precision_ci_low": dp_lo,
        "delta_deferral_precision_ci_high": dp_hi,
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
    }


def run_single_seed(cfg: MSALinearDeferralConfig, seed: int) -> dict[str, Any]:
    ckpt = load_checkpoint(checkpoint_path(cfg.source_experiment_dir, cfg.task, seed))  # type: ignore[arg-type]
    params = ckpt["params"]
    mem_state = extract_nrm_v2_state(ckpt["opt_state"])
    use_associative = resolve_associative_mode(cfg, seed)

    ds_cfg = ClinicalDatasetConfig(
        task=cfg.task,  # type: ignore[arg-type]
        data_dir=cfg.data_dir,  # type: ignore[arg-type]
        max_train=cfg.max_train,
        max_cal=cfg.max_cal,
        max_test=cfg.max_test,
        max_external=cfg.max_external,
    )
    bundle = load_clinical_bundle(ds_cfg)

    x_cal, y_cal, ids_cal = extract_split_arrays(bundle, "calibration")
    x_ext, y_ext, ids_ext = extract_split_arrays(bundle, "external")
    assert len(set(ids_cal.astype(str)) & set(ids_ext.astype(str))) == 0, "calibration/external leakage"

    h_cal, _, H_cal, pred_cal = compute_representations(params, x_cal, num_classes=cfg.num_classes, batch_size=cfg.batch_size)
    h_ext, _, H_ext, pred_ext = compute_representations(params, x_ext, num_classes=cfg.num_classes, batch_size=cfg.batch_size)
    err_cal = (pred_cal != y_cal).astype(int)
    err_ext = (pred_ext != y_ext).astype(int)

    msa_params: dict[str, jnp.ndarray] | None = None
    M_eff: np.ndarray | None = None
    trajectory_bundle: MemoryCheckpointBundle | None = None
    assoc_cfg = cfg.associative_memory

    if use_associative:
        assoc_state, trajectory_bundle, assoc_cfg = load_associative_artifacts(
            cfg.source_experiment_dir,  # type: ignore[arg-type]
            cfg.task,
            seed,
        )
        z_cal, attn_cal = compute_z_from_associative_trajectories(
            h_cal, trajectory_bundle, cfg, assoc_cfg=assoc_cfg
        )
        z_ext, attn_ext = compute_z_from_associative_trajectories(
            h_ext, trajectory_bundle, cfg, assoc_cfg=assoc_cfg
        )
    else:
        M = load_memory_matrix_from_checkpoint(
            ckpt,
            source_dir=cfg.source_experiment_dir,
            task=cfg.task,
            seed=seed,
        )
        M_eff = fixed_memory_projection(
            M,
            proj_dim=cfg.memory_proj_dim,
            seed=seed + 7,
            block_size=cfg.projection_block_size,
        )
        del M
        msa_params, _ = train_msa_on_calibration(h_cal, H_cal, err_cal, jnp.asarray(M_eff), cfg, seed=seed)
        z_cal, attn_cal = compute_z_and_attention(h_cal, jnp.asarray(M_eff), msa_params, cfg)
        z_ext, attn_ext = compute_z_and_attention(h_ext, jnp.asarray(M_eff), msa_params, cfg)

    models = fit_linear_scorers(H_cal, z_cal, err_cal)
    cal_df = pd.DataFrame({"H": H_cal, "error": err_cal})
    ext_df = pd.DataFrame({"H": H_ext, "error": err_ext})
    for j in range(z_ext.shape[1]):
        cal_df[f"z_{j}"] = z_cal[:, j]
        ext_df[f"z_{j}"] = z_ext[:, j]

    probs: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    metrics_primary: dict[str, dict[str, Any]] = {}
    cal_probs_store: dict[str, np.ndarray] = {}
    for name in ("H", "N", "H+N"):
        model, cols = models[name]
        p_cal = deferral_probability(model, cols, cal_df)
        p_ext = deferral_probability(model, cols, ext_df)
        cal_probs_store[name] = p_cal
        thr = calibrate_deferral_threshold(p_cal, target_deferral_rate=cfg.target_deferral_rate)
        probs[name] = p_ext
        thresholds[name] = thr
        m = deferral_metrics(err_ext, p_ext, thr)
        m.update(classification_metrics(err_ext, p_ext))
        m["scorer"] = name
        m["seed"] = seed
        metrics_primary[name] = m

    z_perm = permute_z_within_h_bins(H_ext, z_ext, n_bins=cfg.n_h_bins_perm, seed=cfg.permutation_seed + seed)
    perm_df = ext_df.copy()
    for j in range(z_perm.shape[1]):
        perm_df[f"z_{j}"] = z_perm[:, j]
    model_hn, cols_hn = models["H+N"]
    w = model_hn.named_steps["clf"].coef_.ravel()
    b = float(model_hn.named_steps["clf"].intercept_[0])
    scaler = model_hn.named_steps["scaler"]
    X_scaled = scaler.transform(perm_df[cols_hn])
    logit_perm = b + X_scaled @ w
    p_perm = 1.0 / (1.0 + np.exp(-logit_perm))
    thr_perm = thresholds["H+N"]
    metrics_perm = deferral_metrics(err_ext, p_perm, thr_perm)
    metrics_perm["scorer"] = "H+N_perm"
    metrics_perm["seed"] = seed

    boot_df = bootstrap_paired(
        err_ext, probs["H"], probs["H+N"], thresholds["H"], thresholds["H+N"],
        n_bootstrap=cfg.n_bootstrap, seed=cfg.bootstrap_seed + seed,
    )
    auroc_h = classification_metrics(err_ext, probs["H"])["auroc"]
    auroc_hn = classification_metrics(err_ext, probs["H+N"])["auroc"]
    boot_df["delta_auroc"] = auroc_hn - auroc_h

    if attn_ext.ndim == 3:
        topk_attn_idx = np.argsort(-attn_ext, axis=-1)[:, :, :3]
    elif attn_ext.ndim == 2:
        topk_attn_idx = np.argsort(-attn_ext, axis=-1)[:, :3]
    else:
        topk_attn_idx = np.empty((0, 0), dtype=np.int64)

    per_sample = pd.DataFrame(
        {
            "seed": seed,
            "sample_id": ids_ext,
            "label": y_ext,
            "prediction": pred_ext,
            "error": err_ext,
            "H": H_ext,
            "defer_prob_H": probs["H"],
            "defer_prob_HN": probs["H+N"],
            "defer_prob_HN_perm": p_perm,
            "deferred_H": (probs["H"] >= thresholds["H"]).astype(int),
            "deferred_HN": (probs["H+N"] >= thresholds["H+N"]).astype(int),
            "deferred_N": (probs["N"] >= thresholds["N"]).astype(int),
        }
    )
    for j in range(z_ext.shape[1]):
        per_sample[f"z_{j}"] = z_ext[:, j]
    per_sample["defer_prob_N"] = deferral_probability(models["N"][0], models["N"][1], ext_df)

    return {
        "seed": seed,
        "metrics": metrics_primary,
        "metrics_perm": metrics_perm,
        "metrics_real_hn": metrics_primary["H+N"],
        "boot_df": boot_df,
        "per_sample": per_sample,
        "attn_ext": attn_ext,
        "topk_attn_idx": topk_attn_idx,
        "M_eff": M_eff,
        "msa_params": msa_params,
        "trajectory_bundle": trajectory_bundle,
        "use_associative_memory": use_associative,
        "thresholds": thresholds,
        "memory_summary": memory_state_summary(mem_state),
        "checkpoint": str(checkpoint_path(cfg.source_experiment_dir, cfg.task, seed)),  # type: ignore
    }


def _plot_all(cfg: MSALinearDeferralConfig, pooled: dict[str, Any], out_dir: Path) -> None:
    fig_dir = out_dir / "plots"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ps = pooled["per_sample"]
    mh, mhn = pooled["metrics"]["H"], pooled["metrics"]["H+N"]

    coverages = np.linspace(0.5, 1.0, 11)
    risks_h, risks_hn = [], []
    for cov in coverages:
        dr = 1 - cov
        for name, arr, risks in [("H", ps["defer_prob_H"].to_numpy(), risks_h), ("H+N", ps["defer_prob_HN"].to_numpy(), risks_hn)]:
            thr = calibrate_deferral_threshold(arr, target_deferral_rate=dr)
            m = deferral_metrics(ps["error"].to_numpy(), arr, thr)
            risks.append(m["selective_risk"])

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    ax = axes[0, 0]
    ax.plot(coverages, risks_h, "o-", label="H")
    ax.plot(coverages, risks_hn, "s-", label="H+N")
    ax.set_xlabel("Coverage"); ax.set_ylabel("Selective risk"); ax.set_title("Risk-coverage"); ax.legend()

    y = ps["error"].to_numpy()
    for ax_i, col, title in zip([axes[0, 1], axes[0, 2]], ["defer_prob_H", "defer_prob_HN"], ["ROC H", "ROC H+N"]):
        fpr, tpr, _ = roc_curve(y, ps[col])
        ax_i.plot(fpr, tpr)
        ax_i.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax_i.set_title(title)

    for ax_i, col, title in zip([axes[1, 0], axes[1, 1]], ["defer_prob_H", "defer_prob_HN"], ["PR H", "PR H+N"]):
        prec, rec, _ = precision_recall_curve(y, ps[col])
        ax_i.plot(rec, prec)
        ax_i.set_title(title)

    defer_rates = np.linspace(0.05, 0.5, 10)
    ec_h, ec_hn, dp_h, dp_hn = [], [], [], []
    for dr in defer_rates:
        for col, ec_l, dp_l in [
            ("defer_prob_H", ec_h, dp_h),
            ("defer_prob_HN", ec_hn, dp_hn),
        ]:
            thr = calibrate_deferral_threshold(ps[col].to_numpy(), target_deferral_rate=dr)
            m = deferral_metrics(y, ps[col].to_numpy(), thr)
            ec_l.append(m["error_capture"])
            dp_l.append(m["deferral_precision"])
    axes[1, 2].plot(defer_rates, ec_h, "o-", label="H")
    axes[1, 2].plot(defer_rates, ec_hn, "s-", label="H+N")
    axes[1, 2].set_title("Error capture vs deferral rate"); axes[1, 2].legend()
    axes[1, 3].plot(defer_rates, dp_h, "o-", label="H")
    axes[1, 3].plot(defer_rates, dp_hn, "s-", label="H+N")
    axes[1, 3].set_title("Deferral precision vs deferral rate"); axes[1, 3].legend()

    axes[0, 3].scatter(ps.loc[ps["error"] == 0, "H"], ps.loc[ps["error"] == 0, "z_0"], s=8, alpha=0.3, label="correct")
    axes[0, 3].scatter(ps.loc[ps["error"] == 1, "H"], ps.loc[ps["error"] == 1, "z_0"], s=8, alpha=0.5, label="error")
    axes[0, 3].set_xlabel("H"); axes[0, 3].set_ylabel("z_N'[0]"); axes[0, 3].legend(fontsize=7)

    fig.suptitle("MSA + Linear Deferral — DermaMNIST-E (pooled)")
    fig.tight_layout()
    fig.savefig(fig_dir / "msa_linear_deferral_summary.png", dpi=150)
    plt.close(fig)

    attn = pooled["attn_ext"]
    mean_correct = attn[ps["error"] == 0].mean(axis=(0, 1))
    mean_error = attn[ps["error"] == 1].mean(axis=(0, 1))
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    x = np.arange(attn.shape[-1])
    ax2.bar(x - 0.15, mean_correct, width=0.3, label="correct")
    ax2.bar(x + 0.15, mean_error, width=0.3, label="error")
    ax2.set_xlabel("Memory token index"); ax2.set_ylabel("Mean attention")
    ax2.set_title("Attention over optimization-history tokens")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(fig_dir / "attention_over_memory.png", dpi=150)
    plt.close(fig2)


def metrics_from_deferred_columns(
    errors: np.ndarray,
    defer_probs: np.ndarray,
    deferred: np.ndarray,
) -> dict[str, Any]:
    errors = np.asarray(errors, dtype=int)
    deferred = np.asarray(deferred, dtype=bool)
    accepted = ~deferred
    n = len(errors)
    n_deferred = int(deferred.sum())
    n_accepted = int(accepted.sum())
    n_errors = int(errors.sum())
    n_errors_deferred = int(errors[deferred].sum()) if n_deferred else 0
    r_all = float(errors.mean())
    r_sel = float(errors[accepted].mean()) if n_accepted else float("nan")
    return {
        "n": n,
        "deferral_rate": float(n_deferred / n),
        "coverage": float(n_accepted / n),
        "selective_risk": r_sel,
        "selective_accuracy": float(1.0 - r_sel) if n_accepted else float("nan"),
        "error_rate_deferred": float(errors[deferred].mean()) if n_deferred else float("nan"),
        "deferral_precision": float(n_errors_deferred / n_deferred) if n_deferred else float("nan"),
        "error_capture": float(n_errors_deferred / n_errors) if n_errors else float("nan"),
        "n_deferred": n_deferred,
        "n_errors": n_errors,
        "n_errors_deferred": n_errors_deferred,
        "risk_all": r_all,
        "risk_reduction": float(r_all - r_sel) if n_accepted else float("nan"),
        **classification_metrics(errors, defer_probs),
    }


def compute_pooled_metrics(pooled_ps: pd.DataFrame) -> dict[str, dict[str, Any]]:
    errors = pooled_ps["error"].to_numpy()
    out: dict[str, dict[str, Any]] = {}
    mapping = {
        "H": ("defer_prob_H", "deferred_H"),
        "N": ("defer_prob_N", "deferred_N"),
        "H+N": ("defer_prob_HN", "deferred_HN"),
    }
    for scorer, (prob_col, defer_col) in mapping.items():
        probs = pooled_ps[prob_col].to_numpy()
        deferred = pooled_ps[defer_col].to_numpy().astype(bool)
        m = metrics_from_deferred_columns(errors, probs, deferred)
        m["scorer"] = scorer
        m["seed"] = "pooled"
        out[scorer] = m
    return out


def run_msa_linear_deferral_experiment(cfg: MSALinearDeferralConfig) -> dict[str, Any]:
    cfg = cfg.resolve_paths()
    out_dir = cfg.output_dir
    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "attention_weights").mkdir(exist_ok=True)
    (out_dir / "model").mkdir(exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    seed_runs = [run_single_seed(cfg, s) for s in cfg.seeds]

    seed_rows = []
    for run in seed_runs:
        for scorer in ("H", "N", "H+N"):
            seed_rows.append(run["metrics"][scorer])
    seed_df = pd.DataFrame(seed_rows)

    pooled_ps = pd.concat([r["per_sample"] for r in seed_runs], ignore_index=True)
    per_seed_boot = pd.concat(
        [r["boot_df"].assign(seed=r["seed"]) for r in seed_runs],
        ignore_index=True,
    )
    pooled_boot = bootstrap_pooled_deferred(
        pooled_ps, n_bootstrap=cfg.n_bootstrap, seed=cfg.bootstrap_seed
    )
    pooled_metrics = compute_pooled_metrics(pooled_ps)

    pooled_h = pooled_metrics["H"]
    pooled_hn = pooled_metrics["H+N"]
    per_seed_pass = [
        r["metrics"]["H+N"]["selective_risk"] < r["metrics"]["H"]["selective_risk"] for r in seed_runs
    ]
    perm_pooled = deferral_metrics(
        pooled_ps["error"].to_numpy(),
        pooled_ps["defer_prob_HN_perm"].to_numpy(),
        float(np.median([r["thresholds"]["H+N"] for r in seed_runs])),
    )
    gate = preregistered_verdict(
        pooled_h, pooled_hn, pooled_boot, pooled_hn, perm_pooled, per_seed_pass, ci_level=cfg.ci_level
    )

    delta_rows = []
    for metric in ("selective_risk", "deferral_precision", "error_capture", "auroc"):
        h_vals = seed_df[seed_df["scorer"] == "H"][metric].to_numpy()
        hn_vals = seed_df[seed_df["scorer"] == "H+N"][metric].to_numpy()
        delta_rows.append(
            {
                "metric": metric,
                "delta_HN_minus_H_mean": float(np.mean(hn_vals - h_vals)) if metric != "selective_risk" else float(np.mean(h_vals - hn_vals)),
                "delta_std": float(np.std(hn_vals - h_vals)),
            }
        )
    delta_df = pd.DataFrame(delta_rows)

    if cfg.save_artifacts:
        seed_df.to_csv(out_dir / "seed_results.csv", index=False)
        pd.DataFrame([pooled_metrics["H"], pooled_metrics["N"], pooled_metrics["H+N"]]).to_csv(
            out_dir / "pooled_results.csv", index=False
        )
        per_seed_boot.to_csv(out_dir / "bootstrap_per_seed.csv", index=False)
        pooled_boot.to_csv(out_dir / "bootstrap_results.csv", index=False)
        perm_rows = [r["metrics_perm"] for r in seed_runs]
        pd.DataFrame(perm_rows).to_csv(out_dir / "permutation_results.csv", index=False)
        pooled_ps.to_csv(out_dir / "per_sample_predictions.csv", index=False)
        delta_df.to_csv(out_dir / "selective_prediction_results.csv", index=False)
        seed_df.to_csv(out_dir / "selective_prediction_per_seed.csv", index=False)

        manifest = {
            "task": cfg.task,
            "dataset": "DermaMNIST + DermaMNIST-E external",
            "seeds": list(cfg.seeds),
            "architecture": "ResNet-18 frozen",
            "memory": (
                "associative trajectory (FFT/attention)"
                if cfg.use_associative_memory
                else "V5B long_term (3 levels), frozen"
            ),
            "use_associative_memory": cfg.use_associative_memory,
            "memory_proj_dim": cfg.memory_proj_dim,
            "memory_projection": "blocked Gaussian M @ P, P not stored (seeded, fixed)",
            "projection_block_size": cfg.projection_block_size,
            "msa_heads": cfg.n_heads,
            "msa_d_k": cfg.d_k,
            "msa_d_out": cfg.d_out,
            "target_deferral_rate": cfg.target_deferral_rate,
            "threshold_selection": "calibration only, defer if P(defer) >= tau",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
        }
        (out_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (out_dir / "config.json").write_text(
            json.dumps(
                {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        for run in seed_runs:
            np.savez_compressed(
                out_dir / "attention_weights" / f"attention_seed{run['seed']}.npz",
                attention=run["attn_ext"],
                topk_indices=run["topk_attn_idx"],
            )
            if run.get("msa_params") is not None and run.get("M_eff") is not None:
                with (out_dir / "model" / f"msa_params_seed{run['seed']}.pkl").open("wb") as f:
                    pickle.dump({"msa": run["msa_params"], "M_eff": run["M_eff"]}, f)

        verdict_md = _final_verdict_md(cfg, gate, pooled_h, pooled_hn, seed_df)
        (out_dir / "FINAL_VERDICT.md").write_text(verdict_md, encoding="utf-8")
        (out_dir / "selective_prediction_summary.md").write_text(verdict_md, encoding="utf-8")

        pooled_run = {"per_sample": pooled_ps, "metrics": pooled_metrics, "attn_ext": np.concatenate([r["attn_ext"] for r in seed_runs])}
        if cfg.show_plots:
            _plot_all(cfg, pooled_run, out_dir)

    return {
        "config": cfg,
        "seed_results": seed_df,
        "pooled_metrics": pooled_metrics,
        "bootstrap": pooled_boot,
        "gate": gate,
        "seed_runs": seed_runs,
        "output_dir": out_dir,
    }


def _final_verdict_md(cfg, gate, pooled_h, pooled_hn, seed_df) -> str:
    v = gate["verdict"]
    return "\n".join(
        [
            "# FINAL MSA + Linear Deferral — PRE-REGISTERED VERDICT",
            "",
            f"## VERDICT: **{v}**",
            "",
            "### Primary comparison (H vs H+N, ~20% deferral, pooled external)",
            "",
            f"| Metric | H | H+N |",
            f"|--------|---|-----|",
            f"| Selective risk | {pooled_h['selective_risk']:.4f} | {pooled_hn['selective_risk']:.4f} |",
            f"| Deferral precision | {pooled_h['deferral_precision']:.4f} | {pooled_hn['deferral_precision']:.4f} |",
            f"| Error capture | {pooled_h['error_capture']:.4f} | {pooled_hn['error_capture']:.4f} |",
            f"| AUROC | {pooled_h['auroc']:.4f} | {pooled_hn['auroc']:.4f} |",
            "",
            f"- delta_selective_risk (Risk_H - Risk_HN): {gate['delta_selective_risk_mean']:.4f}",
            f"- 95% CI: [{gate['delta_selective_risk_ci_low']:.4f}, {gate['delta_selective_risk_ci_high']:.4f}]",
            "",
            "### Criteria",
            "",
            f"1. Risk_HN < Risk_H: {gate['criterion1_risk_HN_lower_than_H']}",
            f"2. CI(delta_risk) excludes 0: {gate['criterion2_delta_risk_ci_excludes_zero']}",
            f"3. Consistent across seeds: {gate['criterion3_not_contradicted_across_seeds']}",
            f"4. Beats H+N permutation null: {gate['criterion4_beats_permutation_null']}",
            "",
            "### Interpretation",
            "",
            (
                "Supports that sample-specific optimization-history retrieval improves selective "
                "prediction beyond predictive entropy alone."
                if v == "PASS"
                else "Does not meet pre-registered criteria for selective-prediction improvement."
            ),
            "",
            "Attention weights are diagnostic retrieval evidence, not causal explanation.",
        ]
    )


def print_final_verdict(results: dict[str, Any]) -> None:
    gate = results["gate"]
    ph, phn = results["pooled_metrics"]["H"], results["pooled_metrics"]["H+N"]
    print("=" * 72)
    print(f"FINAL MSA + LINEAR DEFERRAL — VERDICT: {gate['verdict']}")
    print("=" * 72)
    print(f"  Selective risk H:   {ph['selective_risk']:.4f}")
    print(f"  Selective risk H+N: {phn['selective_risk']:.4f}")
    print(f"  Deferral prec H:    {ph['deferral_precision']:.4f}")
    print(f"  Deferral prec H+N:  {phn['deferral_precision']:.4f}")
    print(f"  Error capture H:    {ph['error_capture']:.4f}")
    print(f"  Error capture H+N:  {phn['error_capture']:.4f}")
    print(f"  delta_risk CI excludes 0: {gate['criterion2_delta_risk_ci_excludes_zero']}")
    print("=" * 72)
