"""End-to-end DermaMNIST pipeline for correctness memory (train + deploy + probe).

Implementation equations (audit): optimizer/correctness_memory.py

Mirrors dermamnist_full_validation.ipynb + memory_vector_vs_magnitude_probe.ipynb
without modifying the residual-memory stack.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from optimizer.associative_memory import AssociativeMemoryConfig
from research.common.clinical_datasets import ClinicalDatasetConfig, load_clinical_bundle
from research.common.clinical_training import ClinicalTrainingConfig, checkpoint_path, save_checkpoint
from research.common.correctness_clinical_training import train_one_seed_correctness
from research.common.correctness_deployment_pipeline import run_correctness_deployment_on_split
from research.common.correctness_memory_io import (
    bootstrap_classifier_checkpoint_if_missing,
    correctness_artifacts_exist,
    correctness_memory_exists,
    load_correctness_artifacts,
    recover_correctness_checkpoint_bundle,
)
from research.correctness_memory_fft.probe_experiments.correctness_probe_analysis import run_analysis

TASK = "dermamnist"
OUTPUT_DIR = REPO_ROOT / "research" / "correctness_memory_fft" / "artifacts"
RESIDUAL_OUTPUT_DIR = REPO_ROOT / "research" / "associative_memory_fft" / "artifacts"
DATA_DIR = REPO_ROOT / "data" / "clinical"
FEATURE_CACHE = REPO_ROOT / "research" / "correctness_memory_fft" / "probe_experiments" / "correctness_feature_cache"
SEEDS = (42, 123, 456)
ASSOC_CFG = AssociativeMemoryConfig(use_fft=True, use_attention=False, correctness_z_dim=7)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_CACHE.mkdir(parents=True, exist_ok=True)

bundle = load_clinical_bundle(ClinicalDatasetConfig(task=TASK, data_dir=DATA_DIR))
train_cfg = ClinicalTrainingConfig(seeds=SEEDS, output_dir=OUTPUT_DIR, verbose=1, associative_memory=ASSOC_CFG)

print("=== Part I: train classifier + correctness memory (classifier unchanged) ===")
for seed in SEEDS:
    ckpt = checkpoint_path(OUTPUT_DIR, TASK, seed)
    if correctness_memory_exists(OUTPUT_DIR, TASK, seed):
        recover_correctness_checkpoint_bundle(OUTPUT_DIR, TASK, seed)
        bootstrap_classifier_checkpoint_if_missing(OUTPUT_DIR, RESIDUAL_OUTPUT_DIR, TASK, seed)
    if ckpt.exists() and correctness_artifacts_exist(OUTPUT_DIR, TASK, seed):
        print(f"  seed {seed}: checkpoint + correctness memory exist, skip train")
        continue
    params, opt_state, _, _, summary = train_one_seed_correctness(bundle, seed=seed, cfg=train_cfg)
    save_checkpoint(ckpt, params=params, opt_state=opt_state, task=TASK, seed=seed, cfg=train_cfg, summary=summary)
    print(f"  seed {seed}: trained test_acc={summary.get('test_acc'):.4f}")

print("\n=== Part II: deploy correctness memory features ===")
for seed in SEEDS:
    seed_dir = FEATURE_CACHE / f"seed{seed}"
    cal_path = seed_dir / "cal_features.csv"
    test_path = seed_dir / "test_features_full.csv"
    if cal_path.exists() and test_path.exists():
        print(f"  seed {seed}: feature cache exists, skip deploy")
        continue
    seed_dir.mkdir(parents=True, exist_ok=True)
    with checkpoint_path(OUTPUT_DIR, TASK, seed).open("rb") as f:
        params = pickle.load(f)["params"]
    mem, checkpoints, _cfg = load_correctness_artifacts(OUTPUT_DIR, TASK, seed)
    for split, x, y, ids in [
        ("cal", bundle.x_cal, bundle.y_cal, bundle.sample_ids["cal"]),
        ("test", bundle.x_test, bundle.y_test, bundle.sample_ids["test"]),
    ]:
        _, df = run_correctness_deployment_on_split(
            params,
            x,
            y,
            ids,
            mem,
            num_classes=bundle.num_classes,
        )
        out = cal_path if split == "cal" else test_path
        df.to_csv(out, index=False)
    print(f"  seed {seed}: deployed cal/test features")

print("\n=== Part III: probe analysis (z_combined logistic probe) ===")
summary = run_analysis()
print(
    f"  memory_combined AUROC={summary['mean_auroc_memory_combined']:.4f}  "
    f"H={summary['mean_auroc_H']:.4f}  delta={summary['mean_delta_auroc']:+.4f}"
)
