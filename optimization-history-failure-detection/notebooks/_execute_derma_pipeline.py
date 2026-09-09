"""One-off executor mirroring notebooks/dermamnist_full_validation.ipynb (FAST_MODE)."""
from __future__ import annotations

import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from optimizer.associative_memory import AssociativeMemoryConfig, value_from_logits
from research.common.clinical_datasets import ClinicalDatasetConfig, load_clinical_bundle
from research.common.clinical_training import ClinicalTrainingConfig, checkpoint_path, export_scoring_artifacts_for_nro, save_checkpoint, save_frozen_memory, train_one_seed
from research.common.memory import associative_artifacts_exist, load_associative_artifacts, memory_reconstruction_diagnostics
from research.common.msa_linear_deferral import MSALinearDeferralConfig, run_msa_linear_deferral_experiment
from research.common.nro_final_experiment import NROFinalConfig, run_nro_final_experiment
from research.common.resnet import resnet18_apply, resnet18_features
from research.incremental_memory_deferral.incremental_memory_deferral import IncrementalDeferralConfig, run_incremental_memory_deferral
from research.memory_identification.memory_identification import MemoryIdentificationConfig, run_memory_identification

TASK = "dermamnist"
OUTPUT_DIR = REPO_ROOT / "research" / "associative_memory_fft" / "artifacts"
DATA_DIR = REPO_ROOT / "data" / "clinical"
SEEDS = (42,)
EPOCHS, MAX_TRAIN, MAX_CAL, MAX_TEST, MAX_EXT, N_BOOT = 2, 800, 200, 100, 100, 100
ASSOC_CFG = AssociativeMemoryConfig(use_fft=True, use_attention=False)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
train_cfg = ClinicalTrainingConfig(seeds=SEEDS, epochs=EPOCHS, output_dir=OUTPUT_DIR, verbose=1, associative_memory=ASSOC_CFG)
bundle = load_clinical_bundle(
    ClinicalDatasetConfig(task=TASK, data_dir=DATA_DIR, max_train=MAX_TRAIN, max_cal=MAX_CAL, max_test=MAX_TEST, max_external=MAX_EXT)
)

for seed in SEEDS:
    ckpt = checkpoint_path(OUTPUT_DIR, TASK, seed)
    if not (ckpt.exists() and associative_artifacts_exist(OUTPUT_DIR, TASK, seed)):
        params, opt_state, _, _, summary = train_one_seed(bundle, seed=seed, cfg=train_cfg)
        save_checkpoint(ckpt, params=params, opt_state=opt_state, task=TASK, seed=seed, cfg=train_cfg, summary=summary)
        save_frozen_memory(OUTPUT_DIR / "memory" / f"{TASK}_seed{seed}_memory.npz", opt_state, task=TASK, seed=seed)
        export_scoring_artifacts_for_nro(bundle, params=params, opt_state=opt_state, seed=seed, cfg=train_cfg, output_dir=OUTPUT_DIR)
        print("trained", seed, summary.get("test_acc"))

if not (OUTPUT_DIR / "memory_reconstruction_diagnostics.csv").exists():
    diag_frames = []
    for seed in SEEDS:
        state, _, _ = load_associative_artifacts(OUTPUT_DIR, TASK, seed)
        with checkpoint_path(OUTPUT_DIR, TASK, seed).open("rb") as f:
            params = pickle.load(f)["params"]
        x = bundle.x_train[:32]
        y = bundle.y_train[:32]
        x_j = jnp.asarray(x, dtype=jnp.float32)
        logits = resnet18_apply(params, x_j)
        h = np.asarray(resnet18_features(params, x_j))
        keys = h / (np.linalg.norm(h, axis=-1, keepdims=True) + 1e-8)
        values = np.asarray(value_from_logits(logits, jnp.asarray(y), bundle.num_classes))
        diag = memory_reconstruction_diagnostics(state, keys, values)
        diag["seed"] = seed
        diag_frames.append(diag)
    recon_df = pd.concat(diag_frames, ignore_index=True)
    recon_df.to_csv(OUTPUT_DIR / "memory_reconstruction_diagnostics.csv", index=False)
else:
    recon_df = pd.read_csv(OUTPUT_DIR / "memory_reconstruction_diagnostics.csv")
recon_summary = recon_df.groupby("level")[["mse", "cosine"]].mean()

if not (OUTPUT_DIR / "calibration" / f"{TASK}_seed{SEEDS[0]}.csv").exists():
    for seed in SEEDS:
        with checkpoint_path(OUTPUT_DIR, TASK, seed).open("rb") as f:
            payload = pickle.load(f)
        export_scoring_artifacts_for_nro(
            bundle,
            params=payload["params"],
            opt_state=payload["opt_state"],
            seed=seed,
            cfg=train_cfg,
            output_dir=OUTPUT_DIR,
        )

if not (REPO_ROOT / "research" / "memory_identification" / "artifacts" / "experiment1_per_seed.csv").exists():
    mha_light = MSALinearDeferralConfig(max_train=MAX_TRAIN, max_cal=MAX_CAL, max_test=MAX_TEST, max_external=MAX_EXT)
    run_memory_identification(
        MemoryIdentificationConfig(
            repo_root=REPO_ROOT,
            data_dir=DATA_DIR,
            source_experiment_dir=OUTPUT_DIR,
            task=TASK,
            seeds=SEEDS,
            use_associative_memory=True,
            associative_memory=ASSOC_CFG,
            train_if_checkpoint_missing=False,
            n_bootstrap=N_BOOT,
            n_random_memory=1,
            run_experiment2=False,
            mha=mha_light,
        )
    )

if not (OUTPUT_DIR / "nro_final" / "results.csv").exists():
    run_nro_final_experiment(
        NROFinalConfig(
            repo_root=REPO_ROOT,
            source_experiment_dir=OUTPUT_DIR,
            output_dir=OUTPUT_DIR / "nro_final",
            seeds=SEEDS,
            task=TASK,
            n_bootstrap=N_BOOT,
        )
    )

defer_cfg = MSALinearDeferralConfig(
    repo_root=REPO_ROOT,
    data_dir=DATA_DIR,
    source_experiment_dir=OUTPUT_DIR,
    output_dir=OUTPUT_DIR / "deferral",
    seeds=SEEDS,
    task=TASK,
    use_associative_memory=True,
    associative_memory=ASSOC_CFG,
    max_train=MAX_TRAIN,
    max_cal=MAX_CAL,
    max_test=MAX_TEST,
    max_external=MAX_EXT,
    n_bootstrap=N_BOOT,
).resolve_paths()

if not (OUTPUT_DIR / "deferral" / "msa_linear_deferral_pooled.csv").exists():
    defer_results = run_msa_linear_deferral_experiment(defer_cfg)
else:
    defer_results = {"pooled_primary": pd.read_csv(OUTPUT_DIR / "deferral" / "msa_linear_deferral_pooled.csv").to_dict()}

if not (OUTPUT_DIR / "incremental_deferral" / "incremental_deferral_paired_ci.csv").exists():
    incr_results = run_incremental_memory_deferral(
        IncrementalDeferralConfig(mha=replace(defer_cfg, output_dir=OUTPUT_DIR / "incremental_deferral")).resolve()
    )
else:
    incr_results = {"paired_ci": pd.read_csv(OUTPUT_DIR / "incremental_deferral" / "incremental_deferral_paired_ci.csv").to_dict()}

for label, assoc in [("z_raw", replace(ASSOC_CFG, use_fft=False)), ("z_no_attn", replace(ASSOC_CFG, use_attention=False))]:
    out = OUTPUT_DIR / f"deferral_ablation_{label}" / "msa_linear_deferral_pooled.csv"
    if not out.exists():
        run_msa_linear_deferral_experiment(
            replace(defer_cfg, output_dir=OUTPUT_DIR / f"deferral_ablation_{label}", associative_memory=assoc)
        )

(OUTPUT_DIR / "verdict_draft.json").write_text(
    json.dumps(
        {
            "reconstruction": recon_summary.to_dict(),
            "deferral": defer_results.get("pooled_primary"),
            "incremental": incr_results.get("paired_ci"),
            "fast_mode": True,
            "seeds": SEEDS,
        },
        indent=2,
        default=str,
    ),
    encoding="utf-8",
)
print("DERMA DONE")
