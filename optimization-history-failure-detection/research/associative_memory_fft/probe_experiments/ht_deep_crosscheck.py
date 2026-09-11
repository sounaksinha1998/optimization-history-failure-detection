"""Deep cross-check: H vs h_T gap, entropy recomputation, memory z_j vs cache."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from optimizer.associative_memory import associative_retrieve, normalize_key
from research.associative_memory_fft.probe_experiments import probe_analysis as pa
from research.associative_memory_fft.probe_experiments.ht_linear_probe_dermamnist import HT_COLS
from research.common.clinical_datasets import ClinicalDatasetConfig, load_clinical_bundle
from research.common.clinical_training import checkpoint_path
from research.common.deployment_pipeline import (
    compute_prediction_output,
    compute_representation_and_key,
    normalized_entropy_scalar,
    query_memory_levels,
)
from research.common.memory import load_associative_artifacts
from research.common.resnet import resnet18_probs

TASK = "dermamnist"
SEED = 42
ART = REPO / "research" / "associative_memory_fft" / "artifacts"
DATA = REPO / "data" / "clinical"
PROBE = Path(__file__).resolve().parent
CACHE = PROBE / "feature_cache" / f"seed{SEED}"


def load_test_subsample() -> pd.DataFrame:
    test = pd.read_csv(CACHE / "test_features_full.csv")
    return pa.subsample_test_df(test, sample_size=pa.TEST_SAMPLE_SIZE, seed=pa.RANDOM_SEED + SEED)


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s))


def check_entropy_recompute(bundle, params, test_df: pd.DataFrame) -> dict:
    id_cal = {str(s): i for i, s in enumerate(bundle.sample_ids["test"])}
    diffs_h: list[float] = []
    diffs_norm: list[float] = []
    scores = {
        "cached_H": [],
        "recomputed_H": [],
        "one_minus_maxp": [],
        "neg_margin": [],
        "neg_maxlogit": [],
        "neg_correct_class_prob": [],
    }
    y_list = []
    for _, row in test_df.iterrows():
        i = id_cal[str(row["sample_id"])]
        x = bundle.x_test[i]
        probs = np.asarray(resnet18_probs(params, jnp.asarray(x, dtype=jnp.float32)), dtype=np.float64).reshape(-1)
        pred = compute_prediction_output(probs, num_classes=bundle.num_classes)
        diffs_h.append(abs(row["entropy"] - pred.entropy))
        diffs_norm.append(abs(row["normalized_entropy"] - pred.normalized_entropy))
        sorted_p = np.sort(probs)
        margin = sorted_p[-1] - sorted_p[-2]
        y = int(row["error"])
        y_list.append(y)
        scores["cached_H"].append(float(row["normalized_entropy"]))
        scores["recomputed_H"].append(float(pred.normalized_entropy))
        scores["one_minus_maxp"].append(1.0 - probs.max())
        scores["neg_margin"].append(-margin)
        scores["neg_maxlogit"].append(-probs.max())  # same as 1-maxp for softmax? no - max logit differs
        true_c = int(row["label"])
        scores["neg_correct_class_prob"].append(-probs[true_c])  # USES LABEL - diagnostic only

    y = np.asarray(y_list, dtype=int)
    out = {
        "max_abs_entropy_diff": max(diffs_h),
        "max_abs_normalized_entropy_diff": max(diffs_norm),
        "error_rate": float(y.mean()),
        "accuracy": float(1.0 - y.mean()),
    }
    for name, vals in scores.items():
        if name == "neg_correct_class_prob":
            out[f"auroc_{name}_LEAKAGE"] = auroc(y, np.asarray(vals))
            continue
        out[f"auroc_{name}"] = auroc(y, np.asarray(vals))
    out["corr_cached_H_vs_1_minus_maxp"] = float(
        np.corrcoef(scores["cached_H"], scores["one_minus_maxp"])[0, 1]
    )
    return out


def check_confident_wrong(test_df: pd.DataFrame) -> dict:
    sub = test_df.copy()
    q25 = sub["normalized_entropy"].quantile(0.25)
    low_h = sub[sub["normalized_entropy"] <= q25]
    high_h = sub[sub["normalized_entropy"] > q25]
    return {
        "H_q25_threshold": float(q25),
        "error_rate_all": float(sub["error"].mean()),
        "error_rate_low_H": float(low_h["error"].mean()),
        "error_rate_high_H": float(high_h["error"].mean()),
        "frac_errors_in_low_H": float((sub["error"] == 1).sum() and (low_h["error"] == 1).sum() / (sub["error"] == 1).sum()),
        "n_low_H": int(len(low_h)),
    }


def check_memory_vs_cache(bundle, params, test_df: pd.DataFrame, n: int = 50) -> dict:
    mem, _, _ = load_associative_artifacts(ART, TASK, SEED)
    id_test = {str(s): i for i, s in enumerate(bundle.sample_ids["test"])}
    mag_diffs: list[float] = []
    vec_diffs: list[float] = []
    z_norms: list[float] = []
    h_norms: list[float] = []
    for _, row in test_df.head(n).iterrows():
        i = id_test[str(row["sample_id"])]
        x = bundle.x_test[i]
        rep = compute_representation_and_key(params, x)
        levels = query_memory_levels(rep.key, mem)
        for j, lvl in enumerate(levels, start=1):
            cached_mag = float(row[f"z{j}_magnitude"])
            mag_diffs.append(abs(cached_mag - lvl.magnitude))
            z_norms.append(lvl.magnitude)
            cached_vec = np.array([row[f"z{j}_d{d}"] for d in range(7)], dtype=np.float32)
            vec_diffs.append(float(np.max(np.abs(cached_vec - lvl.response))))
        h_norms.append(float(np.linalg.norm(rep.h)))

    # linearity: z = M k manually for level 1 sample 0
    row0 = test_df.iloc[0]
    i0 = id_test[str(row0["sample_id"])]
    rep0 = compute_representation_and_key(params, bundle.x_test[i0])
    k = jnp.asarray(rep0.key)
    M0 = mem.matrices[0]
    z_manual = np.asarray(associative_retrieve(k, M0), dtype=np.float32)
    z_cache = np.array([row0[f"z1_d{d}"] for d in range(7)], dtype=np.float32)

    return {
        "n_samples_checked": n,
        "max_abs_z_magnitude_diff": max(mag_diffs),
        "max_abs_z_vector_diff": max(vec_diffs),
        "mean_z_magnitude": float(np.mean(z_norms)),
        "mean_h_norm": float(np.mean(h_norms)),
        "z_over_h_norm_ratio": float(np.mean(z_norms) / (np.mean(h_norms) + 1e-12)),
        "sample0_max_diff_z1_manual_vs_cache": float(np.max(np.abs(z_manual - z_cache))),
    }


def check_memory_probe_from_recomputed(bundle, params, test_df: pd.DataFrame, cal_df: pd.DataFrame) -> dict:
    """Recompute z from M,k on cal+test; compare probe AUROC to cached CSV features."""
    mem, _, _ = load_associative_artifacts(ART, TASK, SEED)

    def z_features(df: pd.DataFrame, split: str) -> np.ndarray:
        x_split = bundle.x_cal if split == "cal" else bundle.x_test
        ids = bundle.sample_ids[split]
        id_map = {str(s): i for i, s in enumerate(ids)}
        rows = []
        for sid in df["sample_id"].astype(str):
            i = id_map[sid]
            rep = compute_representation_and_key(params, x_split[i])
            levels = query_memory_levels(rep.key, mem)
            vec = []
            for lvl in levels:
                vec.append(lvl.magnitude)
                vec.extend(lvl.response.tolist())
            rows.append(vec)
        return np.asarray(rows, dtype=np.float64)

    # magnitude-only 4 cols
    def mag_only(Z: np.ndarray) -> np.ndarray:
        return Z[:, [0, 8, 16, 24]]

    cal_df = cal_df.copy()
    test_df = test_df.copy()
    Z_cal = z_features(cal_df, "cal")
    Z_te = z_features(test_df, "test")
    cal_df["_re_z"] = list(Z_cal)
    # use full vector cols from pa.FULL_COLS structure: mag + 7 dims per level
    re_cols = []
    idx = 0
    for j in range(1, 5):
        re_cols.append(f"re_z{j}_mag")
        idx += 1
        for d in range(7):
            re_cols.append(f"re_z{j}_d{d}")
            idx += 1
    for j, col in enumerate(re_cols):
        cal_df[col] = Z_cal[:, j]
        test_df[col] = Z_te[:, j]

    cached_fit = pa.fit_probe(cal_df, pa.FULL_COLS, representation="cached_z")
    cached_ev = pa.eval_probe(cached_fit, test_df)
    re_fit = pa.fit_probe(cal_df, re_cols, representation="recomputed_z")
    re_ev = pa.eval_probe(re_fit, test_df)

    mag_cached = pa.eval_probe(pa.fit_probe(cal_df, pa.MAG_COLS, representation="cm"), test_df)
    mag_re_cols = [f"re_z{j}_mag" for j in range(1, 5)]
    mag_re = pa.eval_probe(pa.fit_probe(cal_df, mag_re_cols, representation="rm"), test_df)

    cache_vs_re = []
    for j in range(1, 5):
        cache_vs_re.append(np.abs(test_df[f"z{j}_magnitude"].to_numpy() - test_df[f"re_z{j}_mag"].to_numpy()))
        for d in range(7):
            cache_vs_re.append(np.abs(test_df[f"z{j}_d{d}"].to_numpy() - test_df[f"re_z{j}_d{d}"].to_numpy()))
    max_feature_diff = float(np.max(np.concatenate(cache_vs_re)))

    h_fit = pa.fit_probe(
        pd.concat(
            [
                cal_df,
                pd.DataFrame(np.load(PROBE / "ht_probe_cache" / f"seed{SEED}" / "cal_ht.npy"), columns=HT_COLS),
            ],
            axis=1,
        ),
        HT_COLS,
        representation="hT",
    )
    h_ev = pa.eval_probe(
        h_fit,
        pd.concat(
            [
                test_df,
                pd.DataFrame(np.load(PROBE / "ht_probe_cache" / f"seed{SEED}" / "test_ht.npy"), columns=HT_COLS),
            ],
            axis=1,
        ),
    )

    return {
        "auroc_z_cached_full": cached_ev["auroc"],
        "auroc_z_recomputed_full": re_ev["auroc"],
        "auroc_z_cached_mag": mag_cached["auroc"],
        "auroc_z_recomputed_mag": mag_re["auroc"],
        "auroc_hT_probe": h_ev["auroc"],
        "auroc_H_cached": pa.eval_entropy(test_df)["auroc"],
        "max_abs_diff_re_vs_cached_features_on_test": max_feature_diff,
    }


def check_label_leakage_upper_bound(test_df: pd.DataFrame, bundle, params) -> dict:
    """Oracle scores that use labels - upper bound if implementation were leaking."""
    id_test = {str(s): i for i, s in enumerate(bundle.sample_ids["test"])}
    oracle_neg_true_prob = []
    y = []
    for _, row in test_df.iterrows():
        i = id_test[str(row["sample_id"])]
        probs = np.asarray(
            resnet18_probs(params, jnp.asarray(bundle.x_test[i], dtype=jnp.float32)), dtype=np.float64
        ).reshape(-1)
        y.append(int(row["error"]))
        oracle_neg_true_prob.append(-probs[int(row["label"])])
    y = np.asarray(y, int)
    return {
        "auroc_neg_p_true_class_ORACLE": auroc(y, np.asarray(oracle_neg_true_prob)),
        "auroc_H_label_free": pa.eval_entropy(test_df)["auroc"],
    }


def main() -> None:
    bundle = load_clinical_bundle(ClinicalDatasetConfig(task=TASK, data_dir=DATA))
    with open(checkpoint_path(ART, TASK, SEED), "rb") as f:
        params = pickle.load(f)["params"]
    cal_df = pd.read_csv(CACHE / "cal_features.csv")
    test_df = load_test_subsample()

    sections = {
        "entropy_recompute_and_scorers": check_entropy_recompute(bundle, params, test_df),
        "confident_wrong_structure": check_confident_wrong(test_df),
        "memory_z_vs_cache": check_memory_vs_cache(bundle, params, test_df),
        "memory_probe_cached_vs_recomputed": check_memory_probe_from_recomputed(
            bundle, params, test_df, cal_df
        ),
        "label_leakage_bounds": check_label_leakage_upper_bound(test_df, bundle, params),
    }

    print("=== Deep cross-check (DermaMNIST seed=42, n_test=2000) ===\n")
    for title, result in sections.items():
        print(f"[{title}]")
        for k, v in result.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")
        print()


if __name__ == "__main__":
    main()
