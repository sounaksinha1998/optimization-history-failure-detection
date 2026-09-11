"""Verify BN axis bug and compute corrected h_T probe (single-sample forward)."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from research.associative_memory_fft.probe_experiments import probe_analysis as pa
from research.associative_memory_fft.probe_experiments.ht_linear_probe_dermamnist import HT_COLS, HT_DIM, _id_index
from research.common.clinical_datasets import ClinicalDatasetConfig, load_clinical_bundle
from research.common.clinical_training import checkpoint_path
from research.common.resnet import resnet18_apply, resnet18_features, resnet18_probs

SEED = 42
ART = REPO / "research" / "associative_memory_fft" / "artifacts"
PROBE = Path(__file__).resolve().parent


def extract_ht_single(params, x_split, id_map, sample_ids) -> np.ndarray:
    rows = []
    for sid in sample_ids.astype(str):
        i = id_map[str(sid)]
        h = np.asarray(
            resnet18_features(params, jnp.asarray(x_split[i], dtype=jnp.float32)),
            dtype=np.float32,
        ).reshape(-1)
        rows.append(h)
    return np.stack(rows, axis=0)


def main() -> None:
    bundle = load_clinical_bundle(ClinicalDatasetConfig(task="dermamnist", data_dir=REPO / "data" / "clinical"))
    with open(checkpoint_path(ART, "dermamnist", SEED), "rb") as f:
        params = pickle.load(f)["params"]

    cal = pd.read_csv(PROBE / "feature_cache/seed42/cal_features.csv")
    test_full = pd.read_csv(PROBE / "feature_cache/seed42/test_features_full.csv")
    test = pa.subsample_test_df(test_full, sample_size=2000, seed=pa.RANDOM_SEED + SEED)

    # Compare batch vs single on 5 samples
    id_test = _id_index(bundle.sample_ids["test"])
    sid5 = test["sample_id"].head(5)
    idx5 = [id_test[str(s)] for s in sid5.astype(str)]
    x5 = jnp.asarray(bundle.x_test[idx5], dtype=jnp.float32)
    h_batch = np.asarray(resnet18_features(params, x5), dtype=np.float32)
    h_single = np.stack(
        [
            np.asarray(resnet18_features(params, jnp.asarray(bundle.x_test[i], dtype=jnp.float32)), dtype=np.float32)
            for i in idx5
        ]
    )
    l_batch = np.asarray(resnet18_apply(params, x5), dtype=np.float64)
    l_single = np.stack(
        [
            np.asarray(resnet18_apply(params, jnp.asarray(bundle.x_test[i], dtype=jnp.float32)), dtype=np.float64)
            for i in idx5
        ]
    )

    print("=== BN / batching bug check (seed 42, first 5 test) ===")
    print(f"max |h_batch - h_single|: {np.max(np.abs(h_batch - h_single)):.6f}")
    print(f"max |logits_batch - logits_single|: {np.max(np.abs(l_batch - l_single)):.6f}")

    # Corrected h_T probe (single-sample, matches deploy)
    print("\nExtracting h_T single-sample (slow, correct) ...")
    cal_ht = extract_ht_single(params, bundle.x_cal, _id_index(bundle.sample_ids["cal"]), cal["sample_id"])
    test_ht = extract_ht_single(params, bundle.x_test, id_test, test["sample_id"])

    cal2 = cal.copy()
    test2 = test.copy()
    for j, c in enumerate(HT_COLS):
        cal2[c] = cal_ht[:, j]
        test2[c] = test_ht[:, j]

    ht_fit = pa.fit_probe(cal2, HT_COLS, representation="h_T_single")
    ht_eval = pa.eval_probe(ht_fit, test2)
    h_eval = pa.eval_entropy(test2)
    z_fit = pa.fit_probe(cal2, pa.FULL_COLS, representation="z")
    z_eval = pa.eval_probe(z_fit, test2)

    # Wrong batched cache if present
    batched = np.load(PROBE / "ht_probe_cache/seed42/test_ht.npy")
    pos = {str(s): i for i, s in enumerate(test_full["sample_id"].astype(str))}
    ht_batched_sub = np.stack([batched[pos[str(s)]] for s in test["sample_id"].astype(str)])
    cal_batched = np.load(PROBE / "ht_probe_cache/seed42/cal_ht.npy")
    cal_b = cal.copy()
    test_b = test.copy()
    for j, c in enumerate(HT_COLS):
        cal_b[c] = cal_batched[:, j]
        test_b[c] = ht_batched_sub[:, j]
    ht_fit_b = pa.fit_probe(cal_b, HT_COLS, representation="h_T_batched")
    ht_eval_b = pa.eval_probe(ht_fit_b, test_b)

    print("\n=== Corrected AUROC (seed 42, n_test=2000) ===")
    print(f"  H (entropy):           {h_eval['auroc']:.4f}")
    print(f"  z_combined:            {z_eval['auroc']:.4f}")
    print(f"  h_T probe (CORRECT):   {ht_eval['auroc']:.4f}")
    print(f"  h_T probe (BATCH BUG): {ht_eval_b['auroc']:.4f}")
    print(f"  cal AUROC correct h_T: {ht_fit['cal_auroc']:.4f}")


if __name__ == "__main__":
    main()
