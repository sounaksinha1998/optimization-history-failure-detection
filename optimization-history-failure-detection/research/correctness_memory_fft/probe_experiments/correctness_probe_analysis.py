"""Probe analysis for high-dimensional correctness memory features.

Implementation equations (audit): see optimizer/correctness_memory.py deploy readout.

Compares (same protocol as residual-memory probe + h_T baseline):
    - entropy_H
    - h_T_probe (512-d penultimate logistic, single-sample forward)
    - z_combined (cal-fit logistic on concat z_1..z_4 in R^{4d})
    - cumulative ablation z1 .. z1-z4
    - per-level z_j probes

Protocol: cal-fit logistic (StandardScaler, C=1), 3 seeds, 2000 test subsample.
"""

from __future__ import annotations

import json
import pickle
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from research.associative_memory_fft.probe_experiments.probe_analysis import (
    BOOTSTRAP_SEED,
    LOGISTIC_C,
    N_BOOTSTRAP,
    RANDOM_SEED,
    SEEDS,
    TEST_SAMPLE_SIZE,
    bootstrap_delta_ci,
    bootstrap_seed_ci,
    eval_entropy,
    eval_probe,
    fit_probe,
    subsample_test_df,
)
from research.common.clinical_datasets import ClinicalDatasetConfig, load_clinical_bundle
from research.common.clinical_training import checkpoint_path
from research.common.resnet import resnet18_features

REPO_ROOT = Path(__file__).resolve().parents[3]
PROBE_DIR = Path(__file__).resolve().parent
FEATURE_CACHE = PROBE_DIR / "correctness_feature_cache"
HT_CACHE = PROBE_DIR / "ht_probe_cache"
RESULTS_DIR = PROBE_DIR / "correctness_probe_results"

TASK = "dermamnist"
ARTIFACT_DIR = REPO_ROOT / "research" / "correctness_memory_fft" / "artifacts"
DATA_DIR = REPO_ROOT / "data" / "clinical"

HT_DIM = 512
HT_COLS = [f"h_{i}" for i in range(HT_DIM)]
NUM_LEVELS = 4
DEFAULT_Z_DIM = 7


def level_cols(j: int, *, z_dim: int = DEFAULT_Z_DIM) -> list[str]:
    return [f"z{j}_d{d}" for d in range(z_dim)]


def cumulative_cols(max_level: int, *, z_dim: int = DEFAULT_Z_DIM) -> list[str]:
    cols: list[str] = []
    for j in range(1, max_level + 1):
        cols.extend(level_cols(j, z_dim=z_dim))
    return cols


Z_COMBINED_COLS = cumulative_cols(NUM_LEVELS, z_dim=DEFAULT_Z_DIM)

PHASE3_MODELS = ("entropy_H", "h_T_probe", "z_combined")


def _resolve_seeds(seeds: Sequence[int] | None) -> list[int]:
    return list(SEEDS if seeds is None else seeds)


def _infer_z_dim(df: pd.DataFrame) -> int:
    cols = [c for c in df.columns if c.startswith("z1_d")]
    if not cols:
        raise ValueError("Feature cache missing z1_d* columns; redeploy Part II.")
    return len(cols)


def _id_index(ids: np.ndarray) -> dict[str, int]:
    return {str(s): int(i) for i, s in enumerate(ids)}


def _load_params(seed: int) -> dict[str, Any]:
    path = checkpoint_path(ARTIFACT_DIR, TASK, seed)
    with path.open("rb") as f:
        return pickle.load(f)["params"]


def extract_ht_matrix(
    params: dict[str, Any],
    x_split: np.ndarray,
    id_to_idx: dict[str, int],
    sample_ids: pd.Series,
) -> np.ndarray:
    """Single-sample forward pass — matches deployment (batch norm is not batch-safe)."""
    rows: list[np.ndarray] = []
    for sid in sample_ids.astype(str):
        i = id_to_idx[str(sid)]
        h = np.asarray(
            resnet18_features(params, jnp.asarray(x_split[i], dtype=jnp.float32)),
            dtype=np.float32,
        ).reshape(-1)
        rows.append(h)
    return np.stack(rows, axis=0)


def attach_ht(df: pd.DataFrame, ht: np.ndarray) -> pd.DataFrame:
    ht_df = pd.DataFrame(ht, columns=HT_COLS, index=df.index)
    return pd.concat([df.reset_index(drop=True), ht_df], axis=1)


def load_or_extract_ht(
    seed: int,
    split: str,
    df: pd.DataFrame,
    *,
    params: dict[str, Any],
    bundle,
    force: bool = False,
) -> pd.DataFrame:
    cache_dir = HT_CACHE / f"seed{seed}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split}_ht.npy"
    ids_path = cache_dir / f"{split}_sample_ids.npy"

    sample_ids = df["sample_id"].astype(str).to_numpy()
    if cache_path.exists() and ids_path.exists() and not force:
        cached_ids = np.load(ids_path, allow_pickle=True)
        if np.array_equal(cached_ids, sample_ids):
            return attach_ht(df, np.load(cache_path))

    x_split = bundle.x_cal if split == "cal" else bundle.x_test
    id_to_idx = _id_index(bundle.sample_ids[split])
    t0 = time.perf_counter()
    ht = extract_ht_matrix(params, x_split, id_to_idx, df["sample_id"])
    elapsed = time.perf_counter() - t0
    np.save(cache_path, ht)
    np.save(ids_path, sample_ids)
    print(f"    extracted h_T seed={seed} {split} n={len(df)} in {elapsed:.1f}s")
    return attach_ht(df, ht)


def load_frames(
    seeds: Sequence[int] | None = None,
    *,
    include_ht: bool = True,
    force_ht: bool = False,
    test_sample_size: int = TEST_SAMPLE_SIZE,
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    use_seeds = _resolve_seeds(seeds)
    cal_frames: dict[int, pd.DataFrame] = {}
    test_frames: dict[int, pd.DataFrame] = {}
    bundle = load_clinical_bundle(ClinicalDatasetConfig(task=TASK, data_dir=DATA_DIR)) if include_ht else None

    for seed in use_seeds:
        cal_path = FEATURE_CACHE / f"seed{seed}" / "cal_features.csv"
        test_path = FEATURE_CACHE / f"seed{seed}" / "test_features_full.csv"
        if not cal_path.exists() or not test_path.exists():
            missing = [p for p in (cal_path, test_path) if not p.exists()]
            raise FileNotFoundError(
                "Correctness feature cache missing for "
                f"seed={seed}. Run Part II of "
                "notebooks/dermamnist_correctness_memory_full.ipynb (or "
                "notebooks/_execute_correctness_derma_pipeline.py) first. "
                f"Missing: {', '.join(str(p) for p in missing)}"
            )
        cal_df = pd.read_csv(cal_path)
        test_full = pd.read_csv(test_path)
        if include_ht:
            assert bundle is not None
            params = _load_params(seed)
            cal_df = load_or_extract_ht(seed, "cal", cal_df, params=params, bundle=bundle, force=force_ht)
            test_full = load_or_extract_ht(seed, "test", test_full, params=params, bundle=bundle, force=force_ht)
        cal_frames[seed] = cal_df
        test_frames[seed] = subsample_test_df(
            test_full, sample_size=test_sample_size, seed=RANDOM_SEED + seed
        )
    return cal_frames, test_frames


def select_z_combined_cols(cal_frames: dict[int, pd.DataFrame]) -> list[str]:
    """Return concatenated z_combined feature columns present in the cache."""
    sample_df = next(iter(cal_frames.values()))
    z_dim = _infer_z_dim(sample_df)
    return cumulative_cols(NUM_LEVELS, z_dim=z_dim)


def comparison_summary_table(summary: dict[str, Any]) -> pd.DataFrame:
    """Mean test AUROC/AUPRC across seeds for main probe models."""
    phase3 = pd.DataFrame(summary["phase3_per_seed"])
    rows: list[dict[str, Any]] = []
    for model in PHASE3_MODELS:
        sub = phase3[phase3["model"] == model]
        if sub.empty:
            continue
        rows.append(
            {
                "model": model,
                "mean_auroc": float(sub["auroc"].mean()),
                "mean_auprc": float(sub["auprc"].mean()),
            }
        )
    return pd.DataFrame(rows).set_index("model")


def ablation_summary_table(summary: dict[str, Any]) -> pd.DataFrame:
    ab = pd.DataFrame(summary["ablation_per_seed"])
    return (
        ab.groupby("levels")[["auroc", "auprc"]]
        .mean()
        .rename(columns={"auroc": "mean_auroc", "auprc": "mean_auprc"})
    )


def run_analysis(
    seeds: Sequence[int] | None = None,
    *,
    include_ht: bool = True,
    force_ht: bool = False,
    test_sample_size: int = TEST_SAMPLE_SIZE,
) -> dict[str, Any]:
    use_seeds = _resolve_seeds(seeds)
    cal_frames, test_frames = load_frames(
        use_seeds,
        include_ht=include_ht,
        force_ht=force_ht,
        test_sample_size=test_sample_size,
    )
    z_combined_cols = select_z_combined_cols(cal_frames)

    phase1_rows: list[dict[str, Any]] = []
    phase3_rows: list[dict[str, Any]] = []
    level_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []

    per_seed_delta_z_h: list[float] = []
    per_seed_delta_z_ht: list[float] = []
    per_seed_delta_ht_h: list[float] = []

    pooled_y: list[np.ndarray] = []
    pooled_h: list[np.ndarray] = []
    pooled_ht: list[np.ndarray] = []
    pooled_z: list[np.ndarray] = []

    z_dim = _infer_z_dim(cal_frames[use_seeds[0]])

    for seed in use_seeds:
        cal_df = cal_frames[seed]
        test_df = test_frames[seed].copy()
        h_eval = eval_entropy(test_df)
        z_fit = fit_probe(cal_df, z_combined_cols, representation="z_combined")
        z_eval = eval_probe(z_fit, test_df)
        phase1_rows.append(
            {
                "seed": seed,
                "representation": "z_combined",
                "cal_auroc": z_fit["cal_auroc"],
                "test_auroc": z_eval["auroc"],
                "test_auprc": z_eval["auprc"],
            }
        )

        ht_eval: dict[str, Any] | None = None
        if include_ht:
            ht_fit = fit_probe(cal_df, HT_COLS, representation="h_T")
            ht_eval = eval_probe(ht_fit, test_df)

        phase3_rows.extend(
            [
                {"seed": seed, "model": "entropy_H", "auroc": h_eval["auroc"], "auprc": h_eval["auprc"]},
                {"seed": seed, "model": "z_combined", "auroc": z_eval["auroc"], "auprc": z_eval["auprc"]},
            ]
        )
        if ht_eval is not None:
            phase3_rows.append(
                {"seed": seed, "model": "h_T_probe", "auroc": ht_eval["auroc"], "auprc": ht_eval["auprc"]}
            )
            per_seed_delta_z_ht.append(z_eval["auroc"] - ht_eval["auroc"])
            per_seed_delta_ht_h.append(ht_eval["auroc"] - h_eval["auroc"])

        per_seed_delta_z_h.append(z_eval["auroc"] - h_eval["auroc"])

        pooled_y.append(z_eval["errors"])
        pooled_h.append(h_eval["scores"])
        pooled_z.append(z_eval["scores"])
        if ht_eval is not None:
            pooled_ht.append(ht_eval["scores"])

        for j in range(1, NUM_LEVELS + 1):
            cols = level_cols(j, z_dim=z_dim)
            lev = eval_probe(fit_probe(cal_df, cols, representation=f"level_{j}"), test_df)
            level_rows.append(
                {
                    "seed": seed,
                    "level": j,
                    "auroc": lev["auroc"],
                    "auprc": lev["auprc"],
                    "n_features": len(cols),
                }
            )

        for max_l in range(1, NUM_LEVELS + 1):
            cols = cumulative_cols(max_l, z_dim=z_dim)
            ab = eval_probe(fit_probe(cal_df, cols, representation=f"cumulative_{max_l}"), test_df)
            ablation_rows.append(
                {
                    "seed": seed,
                    "levels": f"z1..z{max_l}",
                    "max_level": max_l,
                    "auroc": ab["auroc"],
                    "auprc": ab["auprc"],
                }
            )

    y_pool = np.concatenate(pooled_y)
    h_pool = np.concatenate(pooled_h)
    z_pool = np.concatenate(pooled_z)

    z_mean = float(np.mean([r["auroc"] for r in phase3_rows if r["model"] == "z_combined"]))
    h_mean = float(np.mean([r["auroc"] for r in phase3_rows if r["model"] == "entropy_H"]))
    ht_mean = float(
        np.mean([r["auroc"] for r in phase3_rows if r["model"] == "h_T_probe"])
    ) if include_ht else float("nan")

    summary: dict[str, Any] = {
        "memory_kind": "correctness_hd",
        "test_sample_size": test_sample_size,
        "z_dim": z_dim,
        "z_combined_dim": z_dim * NUM_LEVELS,
        "z_combined_cols": z_combined_cols,
        "phase3_summary": {
            "mean_auroc_z_combined": z_mean,
            "mean_auroc_H": h_mean,
            "mean_auroc_h_T_probe": ht_mean,
            "mean_delta_z_minus_H": z_mean - h_mean,
            "mean_delta_z_minus_h_T": z_mean - ht_mean if include_ht else float("nan"),
            "mean_delta_h_T_minus_H": ht_mean - h_mean if include_ht else float("nan"),
            "bootstrap_delta_z_minus_H_per_seed": bootstrap_seed_ci(per_seed_delta_z_h),
            "bootstrap_delta_z_minus_H_pooled": bootstrap_delta_ci(y_pool, h_pool, z_pool),
        },
        "phase1_per_seed": phase1_rows,
        "phase3_per_seed": phase3_rows,
        "level_per_seed": level_rows,
        "ablation_per_seed": ablation_rows,
        # Backward-compatible aliases
        "mean_auroc_memory_combined": z_mean,
        "mean_auroc_H": h_mean,
        "mean_delta_auroc": z_mean - h_mean,
        "bootstrap_delta_per_seed": bootstrap_seed_ci(per_seed_delta_z_h),
    }
    if include_ht:
        summary["phase3_summary"]["bootstrap_delta_z_minus_h_T_per_seed"] = bootstrap_seed_ci(
            per_seed_delta_z_ht
        )
        summary["phase3_summary"]["bootstrap_delta_h_T_minus_H_per_seed"] = bootstrap_seed_ci(
            per_seed_delta_ht_h
        )
        if pooled_ht:
            ht_pool = np.concatenate(pooled_ht)
            summary["phase3_summary"]["bootstrap_delta_z_minus_h_T_pooled"] = bootstrap_delta_ci(
                y_pool, ht_pool, z_pool
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "correctness_probe_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(phase3_rows).to_csv(RESULTS_DIR / "phase3_per_seed.csv", index=False)
    pd.DataFrame(level_rows).to_csv(RESULTS_DIR / "level_per_seed.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(RESULTS_DIR / "ablation_per_seed.csv", index=False)
    return summary


if __name__ == "__main__":
    s = run_analysis()
    p3 = s["phase3_summary"]
    print(
        f"z_combined={p3['mean_auroc_z_combined']:.4f}  "
        f"H={p3['mean_auroc_H']:.4f}  "
        f"h_T={p3['mean_auroc_h_T_probe']:.4f}"
    )
