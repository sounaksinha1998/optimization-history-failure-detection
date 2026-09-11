"""h_T / h_proj32 probe for any task with cached probe_features + checkpoints."""

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

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.associative_memory_fft.probe_experiments import probe_analysis as pa
from research.common.clinical_datasets import ClinicalDatasetConfig, ClinicalTask, load_clinical_bundle
from research.common.clinical_training import checkpoint_path
from research.common.resnet import resnet18_apply, resnet18_features
from research.memory_identification.memory_identification import fixed_input_projection

SEEDS = pa.SEEDS
HT_DIM = 512
HT_COLS = [f"h_{i}" for i in range(HT_DIM)]
H_PROJ_DIM = 32
H_PROJ_SEED = 13
DATA_DIR = REPO_ROOT / "data" / "clinical"
PROBE_ROOT = Path(__file__).resolve().parent


def _id_index(ids: np.ndarray) -> dict[str, int]:
    return {str(s): int(i) for i, s in enumerate(ids)}


def _paths(task: ClinicalTask, *, cross_dataset: bool) -> tuple[Path, Path, Path, Path]:
    if cross_dataset:
        base = REPO_ROOT / "research" / "associative_memory_fft" / "cross_dataset"
        feat = base / "probe_features" / task
        ht_cache = PROBE_ROOT / "ht_probe_cache" / task
        out = PROBE_ROOT / "ht_probe_results" / task
        return base, feat, ht_cache, out
    base = REPO_ROOT / "research" / "associative_memory_fft" / "artifacts"
    feat = PROBE_ROOT / "feature_cache"
    ht_cache = PROBE_ROOT / "ht_probe_cache"
    out = PROBE_ROOT / "ht_probe_results"
    return base, feat, ht_cache, out


def extract_ht_single(
    params: dict[str, Any],
    x_split: np.ndarray,
    id_to_idx: dict[str, int],
    sample_ids: pd.Series,
) -> np.ndarray:
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
    return pd.concat(
        [df.reset_index(drop=True), pd.DataFrame(ht, columns=HT_COLS)],
        axis=1,
    )


def load_or_extract_ht(
    *,
    seed: int,
    split: str,
    df: pd.DataFrame,
    params: dict[str, Any],
    bundle,
    ht_cache: Path,
    force: bool,
) -> pd.DataFrame:
    cache_dir = ht_cache / f"seed{seed}"
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
    ht = extract_ht_single(params, x_split, id_to_idx, df["sample_id"])
    print(f"    h_T {split} seed={seed} n={len(df)} in {time.perf_counter()-t0:.1f}s")
    np.save(cache_path, ht)
    np.save(ids_path, sample_ids)
    return attach_ht(df, ht)


def attach_projected_h(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, list[str]]:
    h_proj = fixed_input_projection(df[HT_COLS].to_numpy(dtype=np.float64), proj_dim=H_PROJ_DIM, seed=seed)
    proj_cols = [f"hproj_{j}" for j in range(H_PROJ_DIM)]
    return pd.concat([df, pd.DataFrame(h_proj, columns=proj_cols)], axis=1), proj_cols


def z_feature_cols(num_classes: int) -> tuple[list[str], list[str]]:
    mag = [f"z{j}_magnitude" for j in range(1, 5)]
    full: list[str] = []
    for j in range(1, 5):
        full.append(f"z{j}_magnitude")
        for d in range(num_classes):
            full.append(f"z{j}_d{d}")
    return mag, full


def select_z_cols(cal_frames: dict[int, pd.DataFrame], num_classes: int) -> list[str]:
    mag_cols, full_cols = z_feature_cols(num_classes)
    rows = []
    for seed in SEEDS:
        for rep, cols in [("magnitude", mag_cols), ("full_vector", full_cols)]:
            rows.append({"representation": rep, "cal_auroc": pa.fit_probe(cal_frames[seed], cols, representation=rep)["cal_auroc"]})
    df = pd.DataFrame(rows)
    mag = float(df.loc[df["representation"] == "magnitude", "cal_auroc"].mean())
    full = float(df.loc[df["representation"] == "full_vector", "cal_auroc"].mean())
    return full_cols if full >= mag else mag_cols


def run_task(task: ClinicalTask, *, cross_dataset: bool, force_ht: bool = False) -> dict[str, Any]:
    artifact_dir, feat_root, ht_cache, output_dir = _paths(task, cross_dataset=cross_dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_clinical_bundle(ClinicalDatasetConfig(task=task, data_dir=DATA_DIR))

    cal_frames: dict[int, pd.DataFrame] = {}
    test_frames: dict[int, pd.DataFrame] = {}
    for seed in SEEDS:
        params = pickle.load(open(checkpoint_path(artifact_dir, task, seed), "rb"))["params"]
        cal_path = feat_root / (f"seed{seed}" if cross_dataset else f"seed{seed}") / "cal_features.csv"
        test_path = feat_root / f"seed{seed}" / "test_features_full.csv"
        if not cross_dataset:
            cal_path = feat_root / f"seed{seed}" / "cal_features.csv"
        cal_df = load_or_extract_ht(
            seed=seed, split="cal", df=pd.read_csv(cal_path), params=params, bundle=bundle, ht_cache=ht_cache, force=force_ht
        )
        test_full = load_or_extract_ht(
            seed=seed,
            split="test",
            df=pd.read_csv(test_path),
            params=params,
            bundle=bundle,
            ht_cache=ht_cache,
            force=force_ht,
        )
        cal_frames[seed] = cal_df
        test_frames[seed] = pa.subsample_test_df(test_full, sample_size=pa.TEST_SAMPLE_SIZE, seed=pa.RANDOM_SEED + seed)

    z_cols = select_z_cols(cal_frames, bundle.num_classes)
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        cal_df, hproj_cols = attach_projected_h(cal_frames[seed], seed=seed + H_PROJ_SEED)
        test_df, _ = attach_projected_h(test_frames[seed], seed=seed + H_PROJ_SEED)
        h_eval = pa.eval_entropy(test_df)
        ht_eval = pa.eval_probe(pa.fit_probe(cal_df, HT_COLS, representation="h_T"), test_df)
        hproj_eval = pa.eval_probe(pa.fit_probe(cal_df, hproj_cols, representation="h_proj32"), test_df)
        z_eval = pa.eval_probe(pa.fit_probe(cal_df, z_cols, representation="z_combined"), test_df)
        h512_z_eval = pa.eval_probe(
            pa.fit_probe(cal_df, [*HT_COLS, *z_cols], representation="h512_plus_z"), test_df
        )
        h32_z_eval = pa.eval_probe(
            pa.fit_probe(cal_df, [*hproj_cols, *z_cols], representation="h32_plus_z"), test_df
        )
        for model, ev in [
            ("entropy_H", h_eval),
            ("h_T_probe", ht_eval),
            ("h_proj32_probe", hproj_eval),
            ("z_combined", z_eval),
            ("h512_plus_z", h512_z_eval),
            ("h32_plus_z", h32_z_eval),
        ]:
            rows.append({"seed": seed, "model": model, "test_auroc": ev["auroc"], "test_auprc": ev["auprc"]})

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(output_dir / "ht_probe_per_seed.csv", index=False)
    summary = {
        "task": task,
        "mean_test_auroc": {m: float(per_seed.loc[per_seed["model"] == m, "test_auroc"].mean()) for m in per_seed["model"].unique()},
        "mean_test_auprc": {m: float(per_seed.loc[per_seed["model"] == m, "test_auprc"].mean()) for m in per_seed["model"].unique()},
    }
    (output_dir / "ht_probe_results.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    force = "--force-ht" in sys.argv
    tasks = [a for a in sys.argv[1:] if not a.startswith("-")] or ["dermamnist", "bloodmnist", "organcmnist"]
    for task in tasks:
        cross = task != "dermamnist"
        print(f"\n=== {task} ===")
        s = run_task(task, cross_dataset=cross, force_ht=force)
        m = s["mean_test_auroc"]
        print(
            f"  H: {m['entropy_H']:.4f}  h512: {m['h_T_probe']:.4f}  h32: {m['h_proj32_probe']:.4f}  "
            f"z: {m['z_combined']:.4f}  h512+z: {m['h512_plus_z']:.4f}  h32+z: {m['h32_plus_z']:.4f}"
        )


if __name__ == "__main__":
    main()
