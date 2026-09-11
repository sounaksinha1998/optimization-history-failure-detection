"""Cross-validation checks for ht_linear_probe_dermamnist.py results."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from research.associative_memory_fft.probe_experiments import probe_analysis as pa
from research.associative_memory_fft.probe_experiments.ht_linear_probe_dermamnist import (
    HT_COLS,
    HT_DIM,
    _id_index,
    _load_params,
    extract_ht_matrix,
)
from research.common.clinical_datasets import ClinicalDatasetConfig, load_clinical_bundle
from research.common.deployment_pipeline import deploy_sample, join_ground_truth
from research.common.memory import load_associative_artifacts
from research.common.resnet import resnet18_apply, resnet18_probs

TASK = "dermamnist"
SEED = 42
ART = REPO / "research" / "associative_memory_fft" / "artifacts"
DATA = REPO / "data" / "clinical"
CACHE = Path(__file__).resolve().parent / "feature_cache" / f"seed{SEED}"
HT_CACHE = Path(__file__).resolve().parent / "ht_probe_cache" / f"seed{SEED}"


def _pipe():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=pa.LOGISTIC_C, max_iter=5000, random_state=42)),
        ]
    )


def check_error_labels(cal_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    cal_err = (cal_df["label"].astype(int) != cal_df["predicted_class"].astype(int)).astype(int)
    test_err = (test_df["label"].astype(int) != test_df["predicted_class"].astype(int)).astype(int)
    return {
        "cal_error_match_rate": float((cal_err.values == cal_df["error"].astype(int).values).mean()),
        "test_error_match_rate": float((test_err.values == test_df["error"].astype(int).values).mean()),
        "cal_error_rate": float(cal_df["error"].mean()),
        "test_error_rate": float(test_df["error"].mean()),
    }


def check_ht_vs_deploy(params, bundle, cal_df: pd.DataFrame, n: int = 20) -> dict:
    mem, cps, _ = load_associative_artifacts(ART, TASK, SEED)
    id_to_idx = _id_index(bundle.sample_ids["cal"])
    diffs = []
    for sid in cal_df["sample_id"].astype(str).head(n):
        i = id_to_idx[sid]
        x = bundle.x_cal[i]
        rec = deploy_sample(
            params,
            x,
            mem,
            sample_id=sid,
            num_classes=bundle.num_classes,
            checkpoints=cps,
            include_history=False,
        )
        h_deploy = rec.representation.h
        h_mine = np.asarray(
            __import__(
                "research.common.resnet", fromlist=["resnet18_features"]
            ).resnet18_features(params, jnp.asarray(x, dtype=jnp.float32)),
            dtype=np.float32,
        ).reshape(-1)
        diffs.append(float(np.max(np.abs(h_deploy - h_mine))))
    return {"max_abs_diff_ht_deploy_n": n, "max_diff": max(diffs), "mean_diff": float(np.mean(diffs))}


def check_ht_cache_alignment(cal_df: pd.DataFrame) -> dict:
    ht = np.load(HT_CACHE / "cal_ht.npy")
    ids = np.load(HT_CACHE / "cal_sample_ids.npy", allow_pickle=True)
    df_ids = cal_df["sample_id"].astype(str).to_numpy()
    if not np.array_equal(ids, df_ids):
        return {"ids_match": False}
    attached = cal_df[HT_COLS].to_numpy(dtype=np.float32)
    return {
        "ids_match": True,
        "max_abs_diff_cache_vs_df": float(np.max(np.abs(ht - attached))),
    }


def check_shuffle_null(cal_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    pipe = _pipe()
    x_cal = cal_df[HT_COLS].to_numpy(dtype=np.float64)
    y_cal = cal_df["error"].to_numpy(dtype=int)
    pipe.fit(x_cal, y_cal)
    x_te = test_df[HT_COLS].to_numpy(dtype=np.float64)
    y_te = test_df["error"].to_numpy(dtype=int)
    scores = pipe.predict_proba(x_te)[:, 1]
    rng = np.random.default_rng(0)
    y_shuf = rng.permutation(y_te)
    return {
        "real_test_auroc": float(roc_auc_score(y_te, scores)),
        "shuffled_label_auroc": float(roc_auc_score(y_shuf, scores)),
    }


def check_logit_baselines(cal_df: pd.DataFrame, test_df: pd.DataFrame, params) -> dict:
    """Scores derived label-free from h_T -> logits (no error labels in construction)."""

    def logits_for_df(df: pd.DataFrame, split: str) -> np.ndarray:
        bundle = load_clinical_bundle(ClinicalDatasetConfig(task=TASK, data_dir=DATA))
        x_split = bundle.x_cal if split == "cal" else bundle.x_test
        id_to_idx = _id_index(bundle.sample_ids[split])
        idx = [id_to_idx[str(s)] for s in df["sample_id"].astype(str)]
        logits_list = []
        for start in range(0, len(idx), 64):
            batch_idx = idx[start : start + 64]
            xb = jnp.asarray(x_split[batch_idx], dtype=jnp.float32)
            logits_list.append(np.asarray(resnet18_apply(params, xb), dtype=np.float64))
        return np.concatenate(logits_list, axis=0)

    cal_logits = logits_for_df(cal_df, "cal")
    test_logits = logits_for_df(test_df, "test")
    from scipy.special import softmax

    cal_p = softmax(cal_logits, axis=1)
    test_p = softmax(test_logits, axis=1)
    cal_margin = np.sort(cal_p, axis=1)[:, -1] - np.sort(cal_p, axis=1)[:, -2]
    test_margin = np.sort(test_p, axis=1)[:, -1] - np.sort(test_p, axis=1)[:, -2]
    test_maxp = test_p.max(axis=1)

    y_te = test_df["error"].to_numpy(dtype=int)
    # Higher score = more likely error (like probe). Use 1-max_prob and negative margin.
    return {
        "auroc_1_minus_maxprob_test": float(roc_auc_score(y_te, 1.0 - test_maxp)),
        "auroc_neg_margin_test": float(roc_auc_score(y_te, -test_margin)),
        "auroc_normalized_entropy_test": float(
            roc_auc_score(y_te, test_df["normalized_entropy"].to_numpy(dtype=float))
        ),
        "auroc_ht_probe_test": float(
            pa.eval_probe(pa.fit_probe(cal_df, HT_COLS, representation="h_T"), test_df)["auroc"]
        ),
    }


def check_cv_cal(cal_df: pd.DataFrame) -> dict:
    x = cal_df[HT_COLS].to_numpy(dtype=np.float64)
    y = cal_df["error"].to_numpy(dtype=int)
    pipe = _pipe()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, x, y, cv=cv, scoring="roc_auc")
    return {
        "cv5_cal_auroc_mean": float(scores.mean()),
        "cv5_cal_auroc_std": float(scores.std()),
        "full_cal_fit_auroc": float(
            roc_auc_score(y, _pipe().fit(x, y).predict_proba(x)[:, 1])
        ),
    }


def check_wrong_split_leakage(params, bundle) -> dict:
    """Fit on TRAIN errors (would leak if train seen during classifier training)."""
    cal_df = pd.read_csv(CACHE / "cal_features.csv")
    train_ids = bundle.sample_ids["train"]
    id_to_idx = _id_index(train_ids)
    # sample 1602 train images
    rng = np.random.default_rng(42)
    pick = rng.choice(len(train_ids), size=min(1602, len(train_ids)), replace=False)
    train_sample_ids = train_ids[pick]
    ht = extract_ht_matrix(
        params,
        bundle.x_train,
        id_to_idx,
        pd.Series(train_sample_ids),
    )
    # compute errors on train
    rows = []
    for j, sid in enumerate(train_sample_ids):
        i = id_to_idx[str(sid)]
        x = bundle.x_train[i]
        y = int(bundle.y_train[i])
        logits = np.asarray(resnet18_apply(params, jnp.asarray(x, dtype=jnp.float32)), dtype=np.float64)
        pred = int(logits.argmax())
        rows.append({"error": int(pred != y), **{HT_COLS[k]: ht[j, k] for k in range(HT_DIM)}})
    train_df = pd.DataFrame(rows)
    test_df = pa.subsample_test_df(
        pd.concat(
            [
                pd.read_csv(CACHE / "test_features_full.csv").reset_index(drop=True),
                pd.DataFrame(np.load(HT_CACHE / "test_ht.npy"), columns=HT_COLS),
            ],
            axis=1,
        ),
        sample_size=pa.TEST_SAMPLE_SIZE,
        seed=pa.RANDOM_SEED + SEED,
    )
    fit = pa.fit_probe(train_df, HT_COLS, representation="h_T_train_fit")
    ev = pa.eval_probe(fit, test_df)
    cal_fit = pa.fit_probe(
        pd.concat(
            [
                cal_df.reset_index(drop=True),
                pd.DataFrame(np.load(HT_CACHE / "cal_ht.npy"), columns=HT_COLS),
            ],
            axis=1,
        ),
        HT_COLS,
        representation="h_T_cal_fit",
    )
    cal_ev = pa.eval_probe(cal_fit, test_df)
    return {
        "test_auroc_fit_on_train_errors": ev["auroc"],
        "test_auroc_fit_on_cal_errors": cal_ev["auroc"],
    }


def main() -> None:
    bundle = load_clinical_bundle(ClinicalDatasetConfig(task=TASK, data_dir=DATA))
    params = _load_params(SEED)
    cal_df = pd.read_csv(CACHE / "cal_features.csv")
    test_full = pd.read_csv(CACHE / "test_features_full.csv")
    cal_ht = pd.concat(
        [cal_df.reset_index(drop=True), pd.DataFrame(np.load(HT_CACHE / "cal_ht.npy"), columns=HT_COLS)],
        axis=1,
    )
    test_ht = pd.concat(
        [
            test_full.reset_index(drop=True),
            pd.DataFrame(np.load(HT_CACHE / "test_ht.npy"), columns=HT_COLS),
        ],
        axis=1,
    )
    test_df = pa.subsample_test_df(test_ht, sample_size=pa.TEST_SAMPLE_SIZE, seed=pa.RANDOM_SEED + SEED)

    checks = {}
    checks["error_labels"] = check_error_labels(cal_df, test_df)
    checks["ht_vs_deploy"] = check_ht_vs_deploy(params, bundle, cal_df)
    checks["ht_cache_alignment"] = check_ht_cache_alignment(cal_ht)
    checks["shuffle_null"] = check_shuffle_null(cal_ht, test_df)
    checks["logit_baselines"] = check_logit_baselines(cal_ht, test_df, params)
    checks["cv_cal"] = check_cv_cal(cal_ht)
    checks["split_comparison"] = check_wrong_split_leakage(params, bundle)

    print("=== ht_probe validation (seed=42) ===")
    for name, result in checks.items():
        print(f"\n[{name}]")
        for k, v in result.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
