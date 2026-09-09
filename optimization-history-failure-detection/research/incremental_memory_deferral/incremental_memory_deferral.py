"""Incremental-memory deferral experiment.

Compares whether optimization-history retrieval z(x) adds routing information beyond
the classifier penultimate representation h(x), under the same protocol as the main
MHA deferral experiment.

Methods:
  H
  H+h
  H+h+z_actual
  H+h+z_random

Read-only inputs from ``research/final_experiment/`` (or ``memory_identification/artifacts/``).
Writes only under ``research/incremental_memory_deferral/artifacts/``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from research.common.clinical_datasets import ClinicalDatasetConfig, load_clinical_bundle
from research.common.clinical_training import checkpoint_path, load_checkpoint
from research.common.error_prediction import fit_failure_scorer, predict_error_probability
from research.common.memory import extract_nrm_v2_state, memory_state_summary
from research.common.msa_linear_deferral import (
    MSALinearDeferralConfig,
    bootstrap_ci,
    bootstrap_paired,
    calibrate_deferral_threshold,
    classification_metrics,
    compute_representations,
    compute_z_and_attention,
    deferral_metrics,
    deferral_probability,
    extract_split_arrays,
    fixed_memory_projection,
    load_memory_matrix_from_checkpoint,
    metrics_from_deferred_columns,
    train_msa_on_calibration,
)
from research.memory_identification.memory_identification import random_memory_matrix_matched

METHODS = ("H", "H+h", "H+h+z_actual", "H+h+z_random")
PRIMARY_BASELINE = "H+h"
PRIMARY_PROPOSED = "H+h+z_actual"
RANDOM_CONTROL = "H+h+z_random"
RANDOM_MEMORY_BASE_SEED = 90_001


@dataclass
class IncrementalDeferralConfig:
    mha: MSALinearDeferralConfig | None = None
    random_memory_base_seed: int = RANDOM_MEMORY_BASE_SEED

    def resolve(self, cwd: Path | None = None) -> IncrementalDeferralConfig:
        self.mha = (self.mha or MSALinearDeferralConfig()).resolve_paths(cwd)
        return self


def _resolve_checkpoint_path(cfg: MSALinearDeferralConfig, seed: int) -> Path:
    candidates = [
        cfg.source_experiment_dir / "checkpoints" / f"{cfg.task}_seed{seed}.pkl",
        cfg.repo_root / "research" / "memory_identification" / "artifacts" / "checkpoints" / f"{cfg.task}_seed{seed}.pkl",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No checkpoint for {cfg.task} seed {seed}. Expected one of: {[str(p) for p in candidates]}"
    )


def _memory_source_dir(cfg: MSALinearDeferralConfig, ckpt_path: Path) -> Path:
    if (ckpt_path.parent.parent / "memory").exists():
        return ckpt_path.parent.parent
    return cfg.source_experiment_dir  # type: ignore[return-value]


def _feature_frame(H: np.ndarray, h: np.ndarray, z: np.ndarray | None = None) -> pd.DataFrame:
    parts = {"H": H}
    for j in range(h.shape[1]):
        parts[f"h_{j}"] = h[:, j]
    if z is not None:
        for j in range(z.shape[1]):
            parts[f"z_{j}"] = z[:, j]
    return pd.DataFrame(parts)


def _method_columns(method: str, h_dim: int, z_dim: int) -> list[str]:
    if method == "H":
        return ["H"]
    h_cols = [f"h_{j}" for j in range(h_dim)]
    if method == "H+h":
        return ["H", *h_cols]
    z_cols = [f"z_{j}" for j in range(z_dim)]
    if method in ("H+h+z_actual", "H+h+z_random"):
        return ["H", *h_cols, *z_cols]
    raise ValueError(method)


def _fit_scorer(cal_df: pd.DataFrame, cols: list[str]) -> Pipeline:
    return fit_failure_scorer(cal_df, cols, error_col="error", max_iter=2000)


def _eval_method(
    method: str,
    model: Pipeline,
    cols: list[str],
    cal_df: pd.DataFrame,
    ext_df: pd.DataFrame,
    errors_ext: np.ndarray,
    *,
    target_deferral_rate: float,
) -> dict[str, Any]:
    p_cal = deferral_probability(model, cols, cal_df)
    p_ext = deferral_probability(model, cols, ext_df)
    thr = calibrate_deferral_threshold(p_cal, target_deferral_rate=target_deferral_rate)
    m = deferral_metrics(errors_ext, p_ext, thr)
    m.update(classification_metrics(errors_ext, p_ext))
    m["method"] = method
    m["threshold"] = thr
    return m, p_cal, p_ext


def run_single_seed(cfg: IncrementalDeferralConfig, seed: int) -> dict[str, Any]:
    mha = cfg.mha
    assert mha is not None
    ckpt_path = _resolve_checkpoint_path(mha, seed)
    ckpt = load_checkpoint(ckpt_path)
    params = ckpt["params"]
    mem_state = extract_nrm_v2_state(ckpt["opt_state"])

    M = load_memory_matrix_from_checkpoint(
        ckpt,
        source_dir=_memory_source_dir(mha, ckpt_path),
        task=mha.task,
        seed=seed,
    )
    M_eff_actual = fixed_memory_projection(
        M,
        proj_dim=mha.memory_proj_dim,
        seed=seed + 7,
        block_size=mha.projection_block_size,
    )
    M_rand = random_memory_matrix_matched(M, seed=cfg.random_memory_base_seed + seed)
    M_eff_rand = fixed_memory_projection(
        M_rand,
        proj_dim=mha.memory_proj_dim,
        seed=seed + 7,
        block_size=mha.projection_block_size,
    )

    ds_cfg = ClinicalDatasetConfig(
        task=mha.task,  # type: ignore[arg-type]
        data_dir=mha.data_dir,  # type: ignore[arg-type]
        max_train=mha.max_train,
        max_cal=mha.max_cal,
        max_test=mha.max_test,
        max_external=mha.max_external,
    )
    bundle = load_clinical_bundle(ds_cfg)
    x_cal, y_cal, ids_cal = extract_split_arrays(bundle, "calibration")
    x_ext, y_ext, ids_ext = extract_split_arrays(bundle, "external")

    h_cal, _, H_cal, pred_cal = compute_representations(
        params, x_cal, num_classes=mha.num_classes, batch_size=mha.batch_size
    )
    h_ext, _, H_ext, pred_ext = compute_representations(
        params, x_ext, num_classes=mha.num_classes, batch_size=mha.batch_size
    )
    err_cal = (pred_cal != y_cal).astype(int)
    err_ext = (pred_ext != y_ext).astype(int)

    msa_actual, _ = train_msa_on_calibration(
        h_cal, H_cal, err_cal, jnp.asarray(M_eff_actual), mha, seed=seed
    )
    msa_rand, _ = train_msa_on_calibration(
        h_cal, H_cal, err_cal, jnp.asarray(M_eff_rand), mha, seed=seed + 10_000
    )
    z_cal_actual, _ = compute_z_and_attention(h_cal, jnp.asarray(M_eff_actual), msa_actual, mha)
    z_ext_actual, _ = compute_z_and_attention(h_ext, jnp.asarray(M_eff_actual), msa_actual, mha)
    z_cal_rand, _ = compute_z_and_attention(h_cal, jnp.asarray(M_eff_rand), msa_rand, mha)
    z_ext_rand, _ = compute_z_and_attention(h_ext, jnp.asarray(M_eff_rand), msa_rand, mha)

    cal_h = _feature_frame(H_cal, h_cal)
    ext_h = _feature_frame(H_ext, h_ext)
    cal_hz_actual = _feature_frame(H_cal, h_cal, z_cal_actual)
    ext_hz_actual = _feature_frame(H_ext, h_ext, z_ext_actual)
    cal_hz_rand = _feature_frame(H_cal, h_cal, z_cal_rand)
    ext_hz_rand = _feature_frame(H_ext, h_ext, z_ext_rand)
    for cal_df in (cal_h, cal_hz_actual, cal_hz_rand):
        cal_df["error"] = err_cal
    for ext_df in (ext_h, ext_hz_actual, ext_hz_rand):
        ext_df["error"] = err_ext

    h_dim = h_cal.shape[1]
    z_dim = z_cal_actual.shape[1]

    method_frames = {
        "H": (cal_h[["H", "error"]], ext_h[["H", "error"]]),
        "H+h": (cal_h, ext_h),
        "H+h+z_actual": (cal_hz_actual, ext_hz_actual),
        "H+h+z_random": (cal_hz_rand, ext_hz_rand),
    }

    metrics: dict[str, dict[str, Any]] = {}
    probs_cal: dict[str, np.ndarray] = {}
    probs_ext: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}

    for method in METHODS:
        cal_df, ext_df = method_frames[method]
        cols = _method_columns(method, h_dim, z_dim)
        model = _fit_scorer(cal_df, cols)
        m, p_cal, p_ext = _eval_method(
            method,
            model,
            cols,
            cal_df,
            ext_df,
            err_ext,
            target_deferral_rate=mha.target_deferral_rate,
        )
        m["seed"] = seed
        metrics[method] = m
        probs_cal[method] = p_cal
        probs_ext[method] = p_ext
        thresholds[method] = m["threshold"]

    boot_primary = bootstrap_paired(
        err_ext,
        probs_ext[PRIMARY_BASELINE],
        probs_ext[PRIMARY_PROPOSED],
        thresholds[PRIMARY_BASELINE],
        thresholds[PRIMARY_PROPOSED],
        n_bootstrap=mha.n_bootstrap,
        seed=mha.bootstrap_seed + seed,
    )
    boot_primary = boot_primary.assign(
        comparison=f"{PRIMARY_BASELINE}_vs_{PRIMARY_PROPOSED}",
        seed=seed,
    )
    boot_random = bootstrap_paired(
        err_ext,
        probs_ext[PRIMARY_PROPOSED],
        probs_ext[RANDOM_CONTROL],
        thresholds[PRIMARY_PROPOSED],
        thresholds[RANDOM_CONTROL],
        n_bootstrap=mha.n_bootstrap,
        seed=mha.bootstrap_seed + seed + 1,
    )
    boot_random = boot_random.assign(
        comparison=f"{PRIMARY_PROPOSED}_vs_{RANDOM_CONTROL}",
        seed=seed,
    )

    per_sample = pd.DataFrame(
        {
            "seed": seed,
            "sample_id": ids_ext,
            "error": err_ext,
            "H": H_ext,
        }
    )
    for method in METHODS:
        per_sample[f"defer_prob_{method}"] = probs_ext[method]
        per_sample[f"deferred_{method}"] = (
            probs_ext[method] >= thresholds[method]
        ).astype(int)

    return {
        "seed": seed,
        "metrics": metrics,
        "thresholds": thresholds,
        "per_sample": per_sample,
        "bootstrap_primary": boot_primary,
        "bootstrap_random": boot_random,
        "checkpoint": str(ckpt_path),
        "memory_summary": memory_state_summary(mem_state),
    }


def _pooled_bootstrap(
    pooled_ps: pd.DataFrame,
    method_a: str,
    method_b: str,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    errors = pooled_ps["error"].to_numpy(dtype=int)
    probs_a = pooled_ps[f"defer_prob_{method_a}"].to_numpy(dtype=float)
    probs_b = pooled_ps[f"defer_prob_{method_b}"].to_numpy(dtype=float)
    deferred_a = pooled_ps[f"deferred_{method_a}"].to_numpy(dtype=bool)
    deferred_b = pooled_ps[f"deferred_{method_b}"].to_numpy(dtype=bool)
    n = len(errors)
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        e = errors[idx]
        ma = metrics_from_deferred_columns(e, probs_a[idx], deferred_a[idx])
        mb = metrics_from_deferred_columns(e, probs_b[idx], deferred_b[idx])
        rows.append(
            {
                "bootstrap_id": b,
                "delta_selective_risk": ma["selective_risk"] - mb["selective_risk"],
                "delta_deferral_precision": mb["deferral_precision"] - ma["deferral_precision"],
                "delta_error_capture": mb["error_capture"] - ma["error_capture"],
            }
        )
    return pd.DataFrame(rows)


def run_incremental_memory_deferral(
    cfg: IncrementalDeferralConfig | None = None,
) -> dict[str, Any]:
    cfg = (cfg or IncrementalDeferralConfig()).resolve()
    mha = cfg.mha
    assert mha is not None and mha.output_dir is not None

    out_dir = mha.output_dir or (mha.repo_root / "research" / "incremental_memory_deferral" / "artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_runs = [run_single_seed(cfg, seed) for seed in mha.seeds]

    per_seed_rows: list[dict[str, Any]] = []
    for run in seed_runs:
        for method in METHODS:
            per_seed_rows.append(run["metrics"][method])

    per_seed_df = pd.DataFrame(per_seed_rows)
    pooled_ps = pd.concat([r["per_sample"] for r in seed_runs], ignore_index=True)

    pooled_rows: list[dict[str, Any]] = []
    for method in METHODS:
        probs = pooled_ps[f"defer_prob_{method}"].to_numpy()
        deferred = pooled_ps[f"deferred_{method}"].to_numpy().astype(bool)
        m = metrics_from_deferred_columns(pooled_ps["error"].to_numpy(), probs, deferred)
        m["method"] = method
        m["seed"] = "pooled"
        pooled_rows.append(m)
    pooled_df = pd.DataFrame(pooled_rows)

    boot_primary_per_seed = pd.concat([r["bootstrap_primary"] for r in seed_runs], ignore_index=True)
    boot_random_per_seed = pd.concat([r["bootstrap_random"] for r in seed_runs], ignore_index=True)
    boot_primary_pooled = _pooled_bootstrap(
        pooled_ps,
        PRIMARY_BASELINE,
        PRIMARY_PROPOSED,
        n_bootstrap=mha.n_bootstrap,
        seed=mha.bootstrap_seed,
    )
    boot_random_pooled = _pooled_bootstrap(
        pooled_ps,
        PRIMARY_PROPOSED,
        RANDOM_CONTROL,
        n_bootstrap=mha.n_bootstrap,
        seed=mha.bootstrap_seed + 1,
    )

    def _summarize_boot(boot_df: pd.DataFrame, label: str) -> dict[str, Any]:
        out: dict[str, Any] = {"comparison": label}
        for metric in ("delta_selective_risk", "delta_deferral_precision", "delta_error_capture"):
            mean, lo, hi = bootstrap_ci(boot_df[metric].to_numpy(), mha.ci_level)
            out[f"{metric}_mean"] = mean
            out[f"{metric}_ci_low"] = lo
            out[f"{metric}_ci_high"] = hi
            out[f"{metric}_ci_excludes_zero"] = bool(lo > 0) if metric != "delta_selective_risk" else bool(lo > 0)
        return out

    ci_rows = [
        {**_summarize_boot(boot_primary_per_seed, f"per_seed_{PRIMARY_BASELINE}_vs_{PRIMARY_PROPOSED}"), "scope": "per_seed"},
        {**_summarize_boot(boot_primary_pooled, f"pooled_{PRIMARY_BASELINE}_vs_{PRIMARY_PROPOSED}"), "scope": "pooled"},
        {**_summarize_boot(boot_random_per_seed, f"per_seed_{PRIMARY_PROPOSED}_vs_{RANDOM_CONTROL}"), "scope": "per_seed"},
        {**_summarize_boot(boot_random_pooled, f"pooled_{PRIMARY_PROPOSED}_vs_{RANDOM_CONTROL}"), "scope": "pooled"},
    ]
    ci_df = pd.DataFrame(ci_rows)

    per_seed_df.to_csv(out_dir / "incremental_deferral_per_seed.csv", index=False)
    pooled_df.to_csv(out_dir / "incremental_deferral_pooled.csv", index=False)
    pooled_ps.to_csv(out_dir / "incremental_deferral_per_sample_pooled.csv", index=False)
    ci_df.to_csv(out_dir / "incremental_deferral_paired_ci.csv", index=False)
    boot_primary_per_seed.to_csv(out_dir / "incremental_deferral_bootstrap_per_seed_primary.csv", index=False)
    boot_primary_pooled.to_csv(out_dir / "incremental_deferral_bootstrap_pooled_primary.csv", index=False)

    config_payload = {
        "task": mha.task,
        "seeds": list(mha.seeds),
        "target_deferral_rate": mha.target_deferral_rate,
        "mha_train_steps": mha.msa_train_steps,
        "random_memory_base_seed": cfg.random_memory_base_seed,
        "methods": list(METHODS),
        "primary_hypothesis": f"{PRIMARY_PROPOSED} vs {PRIMARY_BASELINE}",
        "generated": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    hh = pooled_df.set_index("method").loc[PRIMARY_BASELINE]
    hhz = pooled_df.set_index("method").loc[PRIMARY_PROPOSED]
    primary_delta_risk = float(hh["selective_risk"] - hhz["selective_risk"])
    primary_supported = primary_delta_risk > 0 and bool(
        ci_df.loc[
            (ci_df["comparison"] == f"pooled_{PRIMARY_BASELINE}_vs_{PRIMARY_PROPOSED}")
            & (ci_df["scope"] == "pooled"),
            "delta_selective_risk_ci_excludes_zero",
        ].iloc[0]
    )

    # interpretation markdown
    lines = [
        "# Incremental Memory Deferral",
        "",
        f"Generated: {config_payload['generated']}",
        "",
        "## Primary hypothesis",
        f"Does `{PRIMARY_PROPOSED}` improve routing beyond `{PRIMARY_BASELINE}`?",
        "",
        f"Pooled Δ selective risk ({PRIMARY_BASELINE} − {PRIMARY_PROPOSED}): **{primary_delta_risk:+.4f}**",
        f"Supported at 95% CI: **{primary_supported}**",
        "",
        "## Pooled deferral metrics",
        "",
        pooled_df[
            ["method", "selective_risk", "deferral_precision", "error_capture", "deferral_rate", "coverage"]
        ].to_string(index=False),
        "",
        "## Paired bootstrap CIs",
        "",
        ci_df.to_string(index=False),
    ]
    (out_dir / "interpretation.md").write_text("\n".join(lines), encoding="utf-8")

    return {
        "per_seed": per_seed_df,
        "pooled": pooled_df,
        "paired_ci": ci_df,
        "per_sample_pooled": pooled_ps,
        "primary_delta_risk": primary_delta_risk,
        "primary_supported": primary_supported,
        "output_dir": out_dir,
    }


def main() -> None:
    results = run_incremental_memory_deferral()
    print(f"Primary delta selective risk (H+h - H+h+z_actual): {results['primary_delta_risk']:+.4f}")
    print(f"Primary hypothesis supported: {results['primary_supported']}")
    print(f"Artifacts: {results['output_dir']}")


if __name__ == "__main__":
    main()
