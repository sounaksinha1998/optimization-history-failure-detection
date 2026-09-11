"""Standalone DermaMNIST baseline: linear probe on frozen penultimate features h_T.

Compares failure-detection AUROC/AUPRC for:
  - H (normalized entropy)
  - h_T linear probe (cal-fit logistic on 512-d penultimate features)
  - H + h_T (combined logistic)
  - z_combined (primary memory probe; from cached z_j features)

Uses the same leakage-safe protocol as probe_analysis.py:
  fit on calibration only, evaluate on subsampled test (n=2000/seed).
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.associative_memory_fft.probe_experiments import probe_analysis as pa
from research.common.clinical_datasets import ClinicalDatasetConfig, load_clinical_bundle
from research.common.clinical_training import checkpoint_path
from research.common.deployment_pipeline import deploy_sample
from research.common.memory import load_associative_artifacts
from research.common.resnet import resnet18_apply, resnet18_features
from research.memory_identification.memory_identification import fixed_input_projection

TASK = "dermamnist"
SEEDS = pa.SEEDS
HT_DIM = 512
HT_COLS = [f"h_{i}" for i in range(HT_DIM)]
HT_CACHE = Path(__file__).resolve().parent / "ht_probe_cache"
OUTPUT_DIR = Path(__file__).resolve().parent / "ht_probe_results"
ARTIFACT_DIR = REPO_ROOT / "research" / "associative_memory_fft" / "artifacts"
DATA_DIR = REPO_ROOT / "data" / "clinical"
BATCH_SIZE = 64
H_PROJ_DIM = 32
H_PROJ_SEED = 13


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
    """Single-sample forward pass — must match deploy_sample (batch norm is not batch-safe)."""
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


def attach_projected_h(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, list[str]]:
    h = df[HT_COLS].to_numpy(dtype=np.float64)
    h_proj = fixed_input_projection(h, proj_dim=H_PROJ_DIM, seed=seed)
    proj_cols = [f"hproj_{j}" for j in range(H_PROJ_DIM)]
    proj_df = pd.DataFrame(h_proj, columns=proj_cols, index=df.index)
    return pd.concat([df.reset_index(drop=True), proj_df], axis=1), proj_cols


def logits_matrix(
    params: dict[str, Any],
    bundle,
    split: str,
    sample_ids: pd.Series,
) -> np.ndarray:
    """Single-sample logits — matches deployment (see resnet._batch_norm batching note)."""
    x_split = bundle.x_cal if split == "cal" else bundle.x_test
    id_to_idx = _id_index(bundle.sample_ids[split])
    rows: list[np.ndarray] = []
    for sid in sample_ids.astype(str):
        i = id_to_idx[str(sid)]
        logits = np.asarray(
            resnet18_apply(params, jnp.asarray(x_split[i], dtype=jnp.float32)),
            dtype=np.float64,
        ).reshape(-1)
        rows.append(logits)
    return np.stack(rows, axis=0)


def label_free_logit_scores(logits: np.ndarray) -> dict[str, np.ndarray]:
    probs = softmax(logits, axis=1)
    sorted_p = np.sort(probs, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    return {
        "one_minus_max_prob": 1.0 - probs.max(axis=1),
        "neg_margin": -margin,
    }


def validate_implementation(
    seed: int,
    params: dict[str, Any],
    bundle,
    cal_df: pd.DataFrame,
) -> dict[str, Any]:
    mem, cps, _ = load_associative_artifacts(ARTIFACT_DIR, TASK, seed)
    id_to_idx = _id_index(bundle.sample_ids["cal"])
    max_diffs: list[float] = []
    for sid in cal_df["sample_id"].astype(str).head(10):
        i = id_to_idx[sid]
        x = bundle.x_cal[i]
        h_deploy = deploy_sample(
            params, x, mem, sample_id=sid, num_classes=bundle.num_classes, checkpoints=cps, include_history=False
        ).representation.h
        h_mine = np.asarray(resnet18_features(params, jnp.asarray(x, dtype=jnp.float32)), dtype=np.float32).reshape(-1)
        max_diffs.append(float(np.max(np.abs(h_deploy - h_mine))))
    recomputed = (
        cal_df["label"].astype(int).to_numpy() != cal_df["predicted_class"].astype(int)
    ).astype(int)
    return {
        "ht_matches_deploy_max_diff": max(max_diffs),
        "error_label_match_rate": float((recomputed == cal_df["error"].astype(int).to_numpy()).mean()),
    }


def cv_cal_auroc(cal_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    x = cal_df[feature_cols].to_numpy(dtype=np.float64)
    y = cal_df["error"].to_numpy(dtype=int)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=pa.LOGISTIC_C, max_iter=5000, random_state=42)),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, x, y, cv=cv, scoring="roc_auc")
    return {"cv5_mean": float(scores.mean()), "cv5_std": float(scores.std())}


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
            ht = np.load(cache_path)
            return attach_ht(df, ht)

    x_split = bundle.x_cal if split == "cal" else bundle.x_test
    id_to_idx = _id_index(bundle.sample_ids[split])
    t0 = time.perf_counter()
    ht = extract_ht_matrix(params, x_split, id_to_idx, df["sample_id"])
    elapsed = time.perf_counter() - t0
    np.save(cache_path, ht)
    np.save(ids_path, sample_ids)
    print(f"    extracted h_T seed={seed} {split} n={len(df)} in {elapsed:.1f}s")
    return attach_ht(df, ht)


def select_z_cols(cal_frames: dict[int, pd.DataFrame]) -> tuple[str, list[str]]:
    """Pick X_best (full vs magnitude) by mean calibration AUROC, matching probe_analysis."""
    cal_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        cal_df = cal_frames[seed]
        for rep, cols in [("magnitude", pa.MAG_COLS), ("full_vector", pa.FULL_COLS)]:
            fit = pa.fit_probe(cal_df, cols, representation=rep)
            cal_rows.append({"representation": rep, "cal_auroc": fit["cal_auroc"]})
    cal_df_all = pd.DataFrame(cal_rows)
    mag_mean = float(cal_df_all.loc[cal_df_all["representation"] == "magnitude", "cal_auroc"].mean())
    full_mean = float(cal_df_all.loc[cal_df_all["representation"] == "full_vector", "cal_auroc"].mean())
    if full_mean >= mag_mean:
        return "full_vector", pa.FULL_COLS
    return "magnitude", pa.MAG_COLS


def run_experiment(*, force_ht: bool = False) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_clinical_bundle(ClinicalDatasetConfig(task=TASK, data_dir=DATA_DIR))

    cal_frames: dict[int, pd.DataFrame] = {}
    test_frames: dict[int, pd.DataFrame] = {}

    print("Loading feature cache + extracting h_T …")
    for seed in SEEDS:
        params = _load_params(seed)
        cal_path = pa.FEATURE_CACHE / f"seed{seed}" / "cal_features.csv"
        test_path = pa.FEATURE_CACHE / f"seed{seed}" / "test_features_full.csv"
        if not cal_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                f"Missing probe feature cache for seed={seed}. "
                "Run notebooks/memory_vector_vs_magnitude_probe.ipynb deploy cell first."
            )
        cal_df = pd.read_csv(cal_path)
        test_full = pd.read_csv(test_path)
        cal_frames[seed] = load_or_extract_ht(seed, "cal", cal_df, params=params, bundle=bundle, force=force_ht)
        test_full = load_or_extract_ht(seed, "test", test_full, params=params, bundle=bundle, force=force_ht)
        test_frames[seed] = pa.subsample_test_df(
            test_full, sample_size=pa.TEST_SAMPLE_SIZE, seed=pa.RANDOM_SEED + seed
        )

    z_name, z_cols = select_z_cols(cal_frames)
    print(f"  z_combined uses X_best={z_name}")

    validation = validate_implementation(SEEDS[0], _load_params(SEEDS[0]), bundle, cal_frames[SEEDS[0]])
    print(f"  validation seed={SEEDS[0]}: ht_deploy_max_diff={validation['ht_matches_deploy_max_diff']:.0e}")

    per_seed_rows: list[dict[str, Any]] = []
    delta_z_minus_h: list[float] = []
    delta_z_minus_ht: list[float] = []
    delta_z_minus_hht: list[float] = []
    delta_ht_minus_h: list[float] = []

    pooled_y: list[np.ndarray] = []
    pooled_h_scores: list[np.ndarray] = []
    pooled_ht_scores: list[np.ndarray] = []
    pooled_z_scores: list[np.ndarray] = []

    for seed in SEEDS:
        params = _load_params(seed)
        cal_df = cal_frames[seed]
        test_df = test_frames[seed]
        cal_df, hproj_cols = attach_projected_h(cal_df, seed=seed + H_PROJ_SEED)
        test_df, _ = attach_projected_h(test_df, seed=seed + H_PROJ_SEED)

        h_eval = pa.eval_entropy(test_df)
        ht_fit = pa.fit_probe(cal_df, HT_COLS, representation="h_T")
        ht_eval = pa.eval_probe(ht_fit, test_df)
        hproj_fit = pa.fit_probe(cal_df, hproj_cols, representation="h_proj32")
        hproj_eval = pa.eval_probe(hproj_fit, test_df)
        hht_fit = pa.fit_probe(cal_df, ["normalized_entropy", *HT_COLS], representation="H_plus_h_T")
        hht_eval = pa.eval_probe(hht_fit, test_df)
        z_fit = pa.fit_probe(cal_df, z_cols, representation=f"z_combined_{z_name}")
        z_eval = pa.eval_probe(z_fit, test_df)

        test_logits = logits_matrix(params, bundle, "test", test_df["sample_id"])
        logit_scores = label_free_logit_scores(test_logits)
        y_te = test_df["error"].to_numpy(dtype=int)
        lf_maxp_auroc = float(roc_auc_score(y_te, logit_scores["one_minus_max_prob"]))
        lf_margin_auroc = float(roc_auc_score(y_te, logit_scores["neg_margin"]))

        delta_z_minus_h.append(z_eval["auroc"] - h_eval["auroc"])
        delta_z_minus_ht.append(z_eval["auroc"] - ht_eval["auroc"])
        delta_z_minus_hht.append(z_eval["auroc"] - hht_eval["auroc"])
        delta_ht_minus_h.append(ht_eval["auroc"] - h_eval["auroc"])

        pooled_y.append(ht_eval["errors"])
        pooled_h_scores.append(h_eval["scores"])
        pooled_ht_scores.append(ht_eval["scores"])
        pooled_z_scores.append(z_eval["scores"])

        cv_ht = cv_cal_auroc(cal_df, HT_COLS) if seed == SEEDS[0] else None

        for model, ev, cal_auroc, extra in [
            ("entropy_H", h_eval, float("nan"), {}),
            ("label_free_1_minus_max_prob", {"auroc": lf_maxp_auroc, "auprc": float("nan"), "n_eval": len(y_te)}, float("nan"), {}),
            ("label_free_neg_margin", {"auroc": lf_margin_auroc, "auprc": float("nan"), "n_eval": len(y_te)}, float("nan"), {}),
            ("h_T_probe", ht_eval, ht_fit["cal_auroc"], cv_ht or {}),
            ("h_proj32_probe", hproj_eval, hproj_fit["cal_auroc"], {}),
            ("H_plus_h_T", hht_eval, hht_fit["cal_auroc"], {}),
            ("z_combined", z_eval, z_fit["cal_auroc"], {}),
        ]:
            row = {
                "seed": seed,
                "model": model,
                "n_cal": int(ht_fit["n_cal"]) if "probe" in model or model.startswith("H_plus") or model == "z_combined" else int(len(cal_df)),
                "n_test": ev["n_eval"],
                "cal_auroc": cal_auroc,
                "test_auroc": ev["auroc"],
                "test_auprc": ev.get("auprc", float("nan")),
            }
            if extra:
                row["cv5_cal_auroc_mean"] = extra.get("cv5_mean")
                row["cv5_cal_auroc_std"] = extra.get("cv5_std")
            per_seed_rows.append(row)

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(OUTPUT_DIR / "ht_probe_per_seed.csv", index=False)

    def _mean(model: str, col: str) -> float:
        return float(per_seed_df.loc[per_seed_df["model"] == model, col].mean())

    y_pool = np.concatenate(pooled_y)
    summary = {
        "task": TASK,
        "seeds": SEEDS,
        "test_sample_size_per_seed": pa.TEST_SAMPLE_SIZE,
        "ht_dim": HT_DIM,
        "z_best_representation": z_name,
        "mean_test_auroc": {
            "entropy_H": _mean("entropy_H", "test_auroc"),
            "label_free_1_minus_max_prob": _mean("label_free_1_minus_max_prob", "test_auroc"),
            "label_free_neg_margin": _mean("label_free_neg_margin", "test_auroc"),
            "h_T_probe": _mean("h_T_probe", "test_auroc"),
            "h_proj32_probe": _mean("h_proj32_probe", "test_auroc"),
            "H_plus_h_T": _mean("H_plus_h_T", "test_auroc"),
            "z_combined": _mean("z_combined", "test_auroc"),
        },
        "mean_test_auprc": {
            "entropy_H": _mean("entropy_H", "test_auprc"),
            "h_T_probe": _mean("h_T_probe", "test_auprc"),
            "H_plus_h_T": _mean("H_plus_h_T", "test_auprc"),
            "z_combined": _mean("z_combined", "test_auprc"),
        },
        "mean_delta_auroc_vs_H": {
            "h_T_probe": float(np.mean(delta_ht_minus_h)),
            "z_combined": float(np.mean(delta_z_minus_h)),
        },
        "mean_delta_auroc_vs_h_T_probe": {
            "z_combined": float(np.mean(delta_z_minus_ht)),
        },
        "mean_delta_auroc_vs_H_plus_h_T": {
            "z_combined": float(np.mean(delta_z_minus_hht)),
        },
        "bootstrap_delta_auroc": {
            "z_combined_minus_H_per_seed": pa.bootstrap_seed_ci(delta_z_minus_h),
            "z_combined_minus_h_T_per_seed": pa.bootstrap_seed_ci(delta_z_minus_ht),
            "z_combined_minus_H_plus_h_T_per_seed": pa.bootstrap_seed_ci(delta_z_minus_hht),
            "h_T_probe_minus_H_per_seed": pa.bootstrap_seed_ci(delta_ht_minus_h),
            "z_combined_minus_H_pooled": pa.bootstrap_delta_ci(
                y_pool, np.concatenate(pooled_h_scores), np.concatenate(pooled_z_scores)
            ),
            "z_combined_minus_h_T_pooled": pa.bootstrap_delta_ci(
                y_pool, np.concatenate(pooled_ht_scores), np.concatenate(pooled_z_scores)
            ),
        },
        "validation_checks": validation,
        "notes": (
            "Use single-sample ResNet forward (matches deploy). Batched forward with N>1 is INVALID: "
            "resnet._batch_norm averages axes (0,1) on NHWC, mixing the batch dimension. "
            "H and z_combined from feature cache are trustworthy; batched h_T previously was not."
        ),
        "interpretation": (
            "Compare z_combined to label_free_1_minus_max_prob and h_T_probe. "
            "If z_combined < both, memory does not beat representation-based failure scores."
        ),
    }

    (OUTPUT_DIR / "ht_probe_results.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    force = "--force-ht" in sys.argv
    summary = run_experiment(force_ht=force)
    m = summary["mean_test_auroc"]
    print("\n=== DermaMNIST h_T linear probe (mean test AUROC, 3 seeds) ===")
    print(f"  H (entropy):              {m['entropy_H']:.4f}")
    print(f"  1 - max(p) [label-free]:  {m['label_free_1_minus_max_prob']:.4f}")
    print(f"  -margin [label-free]:     {m['label_free_neg_margin']:.4f}")
    print(f"  h_T probe (512-d):        {m['h_T_probe']:.4f}")
    print(f"  h_proj32 probe:           {m['h_proj32_probe']:.4f}")
    print(f"  z_combined:               {m['z_combined']:.4f}")
    print(f"  z vs h_T:                 {summary['mean_delta_auroc_vs_h_T_probe']['z_combined']:+.4f}")
    print(f"\nWrote {OUTPUT_DIR / 'ht_probe_per_seed.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'ht_probe_results.json'}")


if __name__ == "__main__":
    main()
