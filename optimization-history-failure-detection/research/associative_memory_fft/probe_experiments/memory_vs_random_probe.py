"""Controlled probe experiment: z_M = M k_T vs z_R = R k_T (Frobenius-matched random R).

Identical checkpoint, splits, probe, seeds, and feature columns; only the projection
matrix differs. Reports AUROC/AUPRC, seed means, M-R deltas, and bootstrap CIs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.associative_memory_fft.probe_experiments import probe_analysis as pa
from research.associative_memory_fft.probe_experiments.ht_linear_probe_task import (
    SEEDS,
    select_z_cols,
    z_feature_cols,
)
from research.common.clinical_datasets import ClinicalDatasetConfig, load_clinical_bundle
from research.common.memory import load_associative_artifacts

PROBE_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data" / "clinical"
R_BASE_SEED = 9001
TASK_OFFSET = {"dermamnist": 0, "bloodmnist": 10_000, "organcmnist": 20_000}
NUM_LEVELS = 4


def _paths(task: str) -> tuple[Path, Path, Path]:
    cross = task != "dermamnist"
    if cross:
        artifact = REPO_ROOT / "research" / "associative_memory_fft" / "cross_dataset"
        feat = artifact / "probe_features" / task
        ht_cache = PROBE_ROOT / "ht_probe_cache" / task
    else:
        artifact = REPO_ROOT / "research" / "associative_memory_fft" / "artifacts"
        feat = PROBE_ROOT / "feature_cache"
        ht_cache = PROBE_ROOT / "ht_probe_cache"
    return artifact, feat, ht_cache


def _normalize_rows(h: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(h, axis=1, keepdims=True)
    return h / (norms + eps)


def _level_responses(matrices: tuple[np.ndarray, ...], k: np.ndarray) -> np.ndarray:
    n = k.shape[0]
    d_v = matrices[0].shape[0]
    out = np.empty((n, len(matrices), d_v), dtype=np.float64)
    for j, matrix in enumerate(matrices):
        out[:, j, :] = k @ np.asarray(matrix, dtype=np.float64).T
    return out


def _frobenius_matched_random(
    template: tuple[np.ndarray, ...],
    rng: np.random.Generator,
) -> tuple[np.ndarray, ...]:
    mats: list[np.ndarray] = []
    for m in template:
        r = rng.standard_normal(m.shape).astype(np.float64)
        fro_m = float(np.linalg.norm(m))
        fro_r = float(np.linalg.norm(r))
        if fro_r > 0:
            r *= fro_m / fro_r
        mats.append(r)
    return tuple(mats)


def _attach_memory_features(df: pd.DataFrame, responses: np.ndarray, num_classes: int) -> pd.DataFrame:
    out = df.copy()
    for j in range(NUM_LEVELS):
        level = responses[:, j, :]
        out[f"z{j + 1}_magnitude"] = np.linalg.norm(level, axis=1)
        for d in range(num_classes):
            out[f"z{j + 1}_d{d}"] = level[:, d]
    return out


def _load_ht(df: pd.DataFrame, ht_cache: Path, seed: int, split: str) -> np.ndarray:
    cache_dir = ht_cache / f"seed{seed}"
    ht = np.load(cache_dir / f"{split}_ht.npy").astype(np.float64)
    cached_ids = np.load(cache_dir / f"{split}_sample_ids.npy", allow_pickle=True).astype(str)
    id_to_row = {sid: i for i, sid in enumerate(cached_ids)}
    want_ids = df["sample_id"].astype(str).to_numpy()
    idx = np.array([id_to_row[sid] for sid in want_ids], dtype=int)
    return ht[idx]


def _scale_diagnostics(
    m_mats: tuple[np.ndarray, ...],
    r_mats: tuple[np.ndarray, ...],
    k: np.ndarray,
) -> dict[str, float]:
    z_m = _level_responses(m_mats, k)
    z_r = _level_responses(r_mats, k)
    fro_m = [float(np.linalg.norm(m)) for m in m_mats]
    fro_r = [float(np.linalg.norm(r)) for r in r_mats]
    mag_m = np.linalg.norm(z_m, axis=2)
    mag_r = np.linalg.norm(z_r, axis=2)
    return {
        "max_frobenius_rel_diff": float(max(abs(fm - fr) / (fm + 1e-12) for fm, fr in zip(fro_m, fro_r))),
        "mean_response_mag_M": float(mag_m.mean()),
        "mean_response_mag_R": float(mag_r.mean()),
        "response_mag_ratio_M_over_R": float(mag_m.mean() / (mag_r.mean() + 1e-12)),
        "response_mag_std_M": float(mag_m.std()),
        "response_mag_std_R": float(mag_r.std()),
    }


def _verdict(delta_mean: float, ci_low: float, ci_high: float) -> str:
    if ci_low > 0:
        return "M > R: evidence supporting learned history-based construction"
    if ci_high < 0:
        return "M < R: evidence against the proposed mechanism"
    if abs(delta_mean) < 0.01:
        return "M ~ R: memory-specific benefit is not established"
    if delta_mean > 0:
        return "M ~ R (weak positive trend): memory-specific benefit is not established"
    return "M ~ R (weak negative trend): memory-specific benefit is not established"


def run_task(task: str) -> dict[str, Any]:
    artifact_dir, feat_root, ht_cache = _paths(task)
    bundle = load_clinical_bundle(ClinicalDatasetConfig(task=task, data_dir=DATA_DIR))
    num_classes = bundle.num_classes

    seed_data: dict[int, dict[str, Any]] = {}
    per_seed_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, float]] = []
    pooled_y: list[np.ndarray] = []
    pooled_m: list[np.ndarray] = []
    pooled_r: list[np.ndarray] = []

    for seed in SEEDS:
        cal_df_raw = pd.read_csv(feat_root / f"seed{seed}" / "cal_features.csv")
        test_full = pd.read_csv(feat_root / f"seed{seed}" / "test_features_full.csv")
        test_df_raw = pa.subsample_test_df(test_full, sample_size=pa.TEST_SAMPLE_SIZE, seed=pa.RANDOM_SEED + seed)

        h_cal = _load_ht(cal_df_raw, ht_cache, seed, "cal")
        h_test = _load_ht(test_df_raw, ht_cache, seed, "test")
        k_cal = _normalize_rows(h_cal)
        k_test = _normalize_rows(h_test)

        mem, _, _ = load_associative_artifacts(artifact_dir, task, seed)
        m_mats = tuple(np.asarray(m, dtype=np.float64) for m in mem.matrices)
        rng = np.random.default_rng(R_BASE_SEED + TASK_OFFSET.get(task, 0) + seed)
        r_mats = _frobenius_matched_random(m_mats, rng)

        z_m_cal = _level_responses(m_mats, k_cal)
        z_m_test = _level_responses(m_mats, k_test)
        z_r_cal = _level_responses(r_mats, k_cal)
        z_r_test = _level_responses(r_mats, k_test)

        cal_m = _attach_memory_features(cal_df_raw, z_m_cal, num_classes)
        test_m = _attach_memory_features(test_df_raw, z_m_test, num_classes)
        cal_r = _attach_memory_features(cal_df_raw, z_r_cal, num_classes)
        test_r = _attach_memory_features(test_df_raw, z_r_test, num_classes)

        mag_cols, full_cols = z_feature_cols(num_classes)
        cached_mag = cal_df_raw[mag_cols].to_numpy(dtype=np.float64)
        recomputed_mag = z_m_cal[:, :, :].copy()
        recomputed_mag = np.linalg.norm(z_m_cal, axis=2)
        max_mag_diff = float(np.max(np.abs(cached_mag - recomputed_mag)))

        scale_cal = _scale_diagnostics(m_mats, r_mats, k_cal)
        scale_test = _scale_diagnostics(m_mats, r_mats, k_test)
        scale_rows.append({"seed": seed, "split": "cal", **scale_cal})
        scale_rows.append({"seed": seed, "split": "test", **scale_test})

        seed_data[seed] = {
            "cal_m": cal_m,
            "test_m": test_m,
            "cal_r": cal_r,
            "test_r": test_r,
            "max_mag_diff": max_mag_diff,
            "scale_test": scale_test,
        }

    cal_frames_m = {seed: seed_data[seed]["cal_m"] for seed in SEEDS}
    z_cols = select_z_cols(cal_frames_m, num_classes)

    for seed in SEEDS:
        sd = seed_data[seed]
        m_fit = pa.fit_probe(sd["cal_m"], z_cols, representation="z_M")
        m_eval = pa.eval_probe(m_fit, sd["test_m"])
        r_fit = pa.fit_probe(sd["cal_r"], z_cols, representation="z_R")
        r_eval = pa.eval_probe(r_fit, sd["test_r"])

        pooled_y.append(m_eval["errors"])
        pooled_m.append(m_eval["scores"])
        pooled_r.append(r_eval["scores"])

        per_seed_rows.append(
            {
                "seed": seed,
                "n_cal": m_fit["n_cal"],
                "n_test": m_eval["n_eval"],
                "feature_cols": len(z_cols),
                "max_abs_mag_diff_cache_vs_Mk": sd["max_mag_diff"],
                "auroc_M": m_eval["auroc"],
                "auprc_M": m_eval["auprc"],
                "auroc_R": r_eval["auroc"],
                "auprc_R": r_eval["auprc"],
                "delta_auroc_M_minus_R": m_eval["auroc"] - r_eval["auroc"],
                "delta_auprc_M_minus_R": m_eval["auprc"] - r_eval["auprc"],
                **{f"scale_{k}": v for k, v in sd["scale_test"].items()},
            }
        )

    per_seed = pd.DataFrame(per_seed_rows)
    scale_df = pd.DataFrame(scale_rows)

    delta_auroc = per_seed["delta_auroc_M_minus_R"].to_numpy(dtype=float)
    delta_auprc = per_seed["delta_auprc_M_minus_R"].to_numpy(dtype=float)
    auroc_ci = pa.bootstrap_seed_ci(delta_auroc.tolist())
    auprc_ci = pa.bootstrap_seed_ci(delta_auprc.tolist())

    y_pool = np.concatenate(pooled_y)
    m_pool = np.concatenate(pooled_m)
    r_pool = np.concatenate(pooled_r)
    pooled_ci = pa.bootstrap_delta_ci(y_pool, r_pool, m_pool)

    summary = {
        "task": task,
        "num_classes": num_classes,
        "num_levels": NUM_LEVELS,
        "R_scaling": "frobenius_matched_per_level",
        "feature_selection": "same as z_combined (cal AUROC: magnitude vs full_vector on M)",
        "mean_test_auroc": {
            "M": float(per_seed["auroc_M"].mean()),
            "R": float(per_seed["auroc_R"].mean()),
        },
        "std_test_auroc": {
            "M": float(per_seed["auroc_M"].std(ddof=0)),
            "R": float(per_seed["auroc_R"].std(ddof=0)),
        },
        "mean_test_auprc": {
            "M": float(per_seed["auprc_M"].mean()),
            "R": float(per_seed["auprc_R"].mean()),
        },
        "std_test_auprc": {
            "M": float(per_seed["auprc_M"].std(ddof=0)),
            "R": float(per_seed["auprc_R"].std(ddof=0)),
        },
        "delta_M_minus_R": {
            "auroc_mean": float(delta_auroc.mean()),
            "auroc_std": float(delta_auroc.std(ddof=0)),
            "auprc_mean": float(delta_auprc.mean()),
            "auprc_std": float(delta_auprc.std(ddof=0)),
        },
        "bootstrap_ci_auroc_delta_per_seed": auroc_ci,
        "bootstrap_ci_auprc_delta_per_seed": auprc_ci,
        "bootstrap_ci_auroc_delta_pooled_test": pooled_ci,
        "scale_check": {
            "max_frobenius_rel_diff_mean": float(scale_df["max_frobenius_rel_diff"].mean()),
            "response_mag_ratio_M_over_R_mean": float(scale_df["response_mag_ratio_M_over_R"].mean()),
            "response_mag_ratio_M_over_R_std": float(scale_df["response_mag_ratio_M_over_R"].std(ddof=0)),
        },
        "verdict": _verdict(
            float(delta_auroc.mean()),
            auroc_ci["ci_low"],
            auroc_ci["ci_high"],
        ),
        "per_seed": per_seed_rows,
    }
    return summary


def main() -> None:
    tasks = [a for a in sys.argv[1:] if not a.startswith("-")] or ["dermamnist", "bloodmnist", "organcmnist"]
    out_dir = PROBE_ROOT / "ht_probe_results" / "memory_vs_random"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict[str, Any]] = []
    print("Controlled probe: z_M = M k_T  vs  z_R = R k_T  (Frobenius-matched R)\n")
    for task in tasks:
        summary = run_task(task)
        all_summaries.append(summary)
        m = summary["mean_test_auroc"]
        d = summary["delta_M_minus_R"]
        ci = summary["bootstrap_ci_auroc_delta_per_seed"]
        sc = summary["scale_check"]
        print(f"=== {task} ===")
        print(f"  AUROC  M: {m['M']:.4f} +/- {summary['std_test_auroc']['M']:.4f}")
        print(f"  AUROC  R: {m['R']:.4f} +/- {summary['std_test_auroc']['R']:.4f}")
        print(f"  AUPRC  M: {summary['mean_test_auprc']['M']:.4f} +/- {summary['std_test_auprc']['M']:.4f}")
        print(f"  AUPRC  R: {summary['mean_test_auprc']['R']:.4f} +/- {summary['std_test_auprc']['R']:.4f}")
        print(f"  dAUROC (M-R): {d['auroc_mean']:+.4f} +/- {d['auroc_std']:.4f}  "
              f"95% CI [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]")
        print(f"  Scale: ||M||_F ~= ||R||_F (max rel diff {sc['max_frobenius_rel_diff_mean']:.2e}); "
              f"response mag ratio M/R = {sc['response_mag_ratio_M_over_R_mean']:.3f}")
        print(f"  Verdict: {summary['verdict']}\n")

    (out_dir / "memory_vs_random_probe.json").write_text(json.dumps(all_summaries, indent=2))
    pd.concat(
        [
            pd.DataFrame(s["per_seed"]).assign(task=s["task"])
            for s in all_summaries
        ]
    ).to_csv(out_dir / "memory_vs_random_per_seed.csv", index=False)
    print(f"Wrote {out_dir / 'memory_vs_random_probe.json'}")


if __name__ == "__main__":
    main()
