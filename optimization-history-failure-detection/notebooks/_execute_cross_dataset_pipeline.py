"""One-off executor mirroring notebooks/bloodmnist_organcmnist_probe.ipynb (FAST_MODE)."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from optimizer.associative_memory import AssociativeMemoryConfig
from research.common.clinical_datasets import ClinicalDatasetConfig, load_clinical_bundle
from research.common.clinical_training import ClinicalTrainingConfig, checkpoint_path, save_checkpoint, save_frozen_memory, train_one_seed
from research.common.deployment_pipeline import run_deployment_on_split
from research.common.memory import associative_artifacts_exist, load_associative_artifacts

TASKS = ("bloodmnist", "organcmnist")
OUTPUT_DIR = REPO_ROOT / "research" / "associative_memory_fft" / "cross_dataset"
PROBE_DIR = OUTPUT_DIR / "probe_results"
FEATURE_ROOT = OUTPUT_DIR / "probe_features"
DATA_DIR = REPO_ROOT / "data" / "clinical"
SEEDS = (42,)
EPOCHS, MAX_TRAIN, MAX_CAL, MAX_TEST, MAX_EXT = 2, 800, 200, 400, 400
# Full run mirrors DermaMNIST caps: train=6408, cal=1602, test=2005
DERMA_TRAIN, DERMA_CAL, DERMA_TEST = 6408, 1602, 2005
LOGISTIC_C = 1.0
ASSOC_CFG = AssociativeMemoryConfig(use_fft=True, use_attention=False)


def mag_cols() -> list[str]:
    return [f"z{j}_magnitude" for j in range(1, 5)]


def full_vector_cols(num_classes: int) -> list[str]:
    cols: list[str] = []
    for j in range(1, 5):
        cols.extend([f"z{j}_d{d}" for d in range(num_classes)])
    return cols


def fit_probe(cal_df: pd.DataFrame, feature_cols: list[str]) -> Any:
    mask = cal_df[feature_cols + ["error"]].notna().all(axis=1).to_numpy()
    x = cal_df.loc[mask, feature_cols].to_numpy(dtype=float)
    y = cal_df.loc[mask, "error"].to_numpy(dtype=int)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=LOGISTIC_C, max_iter=5000, random_state=42)),
        ]
    )
    pipe.fit(x, y)
    return pipe


def eval_scores(df: pd.DataFrame, feature_cols: list[str], pipe: Any) -> tuple[float, float]:
    mask = df[feature_cols + ["error"]].notna().all(axis=1).to_numpy()
    x = df.loc[mask, feature_cols].to_numpy(dtype=float)
    y = df.loc[mask, "error"].to_numpy(dtype=int)
    if len(set(y)) < 2:
        return float("nan"), float("nan")
    scores = pipe.predict_proba(x)[:, 1]
    return float(roc_auc_score(y, scores)), float(average_precision_score(y, scores))


def eval_h(df: pd.DataFrame) -> tuple[float, float]:
    mask = df[["normalized_entropy", "error"]].notna().all(axis=1).to_numpy()
    y = df.loc[mask, "error"].to_numpy(dtype=int)
    h = df.loc[mask, "normalized_entropy"].to_numpy(dtype=float)
    if len(set(y)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, h)), float(average_precision_score(y, h))


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROBE_DIR.mkdir(parents=True, exist_ok=True)
train_cfg = ClinicalTrainingConfig(seeds=SEEDS, epochs=EPOCHS, output_dir=OUTPUT_DIR, verbose=1, associative_memory=ASSOC_CFG)

rows: list[dict[str, Any]] = []
for task in TASKS:
    bundle = load_clinical_bundle(
        ClinicalDatasetConfig(
            task=task,
            data_dir=DATA_DIR,
            max_train=MAX_TRAIN,
            max_cal=MAX_CAL,
            max_test=MAX_TEST,
            max_external=MAX_EXT,
        )
    )
    for seed in SEEDS:
        ckpt = checkpoint_path(OUTPUT_DIR, task, seed)
        if not ckpt.exists():
            params, opt_state, _, _, summary = train_one_seed(bundle, seed=seed, cfg=train_cfg)
            save_checkpoint(ckpt, params=params, opt_state=opt_state, task=task, seed=seed, cfg=train_cfg, summary=summary)
            save_frozen_memory(OUTPUT_DIR / "memory" / f"{task}_seed{seed}_memory.npz", opt_state, task=task, seed=seed)
            print("trained", task, seed, summary.get("test_acc"))

        with open(ckpt, "rb") as f:
            params = pickle.load(f)["params"]
        mem_state, checkpoints, _ = load_associative_artifacts(OUTPUT_DIR, task, seed)
        _, cal_df = run_deployment_on_split(
            params,
            bundle.x_cal,
            bundle.y_cal,
            bundle.sample_ids["cal"],
            mem_state,
            checkpoints,
            num_classes=bundle.num_classes,
        )
        _, test_df = run_deployment_on_split(
            params,
            bundle.x_test,
            bundle.y_test,
            bundle.sample_ids["test"],
            mem_state,
            checkpoints,
            num_classes=bundle.num_classes,
        )

        mag_pipe = fit_probe(cal_df, mag_cols())
        full_pipe = fit_probe(cal_df, full_vector_cols(bundle.num_classes))
        mag_auroc, _ = eval_scores(cal_df, mag_cols(), mag_pipe)
        full_auroc, _ = eval_scores(cal_df, full_vector_cols(bundle.num_classes), full_pipe)
        best_cols = full_vector_cols(bundle.num_classes) if full_auroc >= mag_auroc else mag_cols()
        z_pipe = fit_probe(cal_df, best_cols)
        z_auroc, z_auprc = eval_scores(test_df, best_cols, z_pipe)
        h_auroc, h_auprc = eval_h(test_df)
        rows.append(
            {
                "task": task,
                "seed": seed,
                "auroc_H": h_auroc,
                "auroc_z_combined": z_auroc,
                "auprc_H": h_auprc,
                "auprc_z_combined": z_auprc,
            }
        )

pd.DataFrame(rows).to_csv(PROBE_DIR / "fast_smoke.csv", index=False)
(PROBE_DIR / "fast_smoke.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print("CROSS PROBE DONE")
