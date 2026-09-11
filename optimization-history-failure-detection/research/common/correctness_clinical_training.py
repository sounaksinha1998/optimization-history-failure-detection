"""Classifier training with correctness-memory side channel (classifier unchanged).

Implementation equations (audit): see optimizer/correctness_memory.py.

Only the memory block differs from clinical_training.train_one_seed:
    c_t = 1[y_hat_t = y_t], u_t = c_t P k_t in R^{d_v}
    M^{(j)} in R^{d_v x d_k}  (default d_v=7, d_k=512)
    assoc_mem.update(k_t, c_t)

Memory keys/targets use single-sample ResNet forwards at pre-update theta_t
(same BN path as deployment). Batch-level delta aggregation is unchanged.
"""

from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from optimizer.associative_memory import AssociativeMemoryConfig, normalize_key
from optimizer.correctness_memory import CorrectnessMemory, CorrectnessMemoryState, correctness_from_logits, expected_matrix_shape
from research.common.clinical_datasets import ClinicalBundle
from research.common.clinical_log import clinical_log
from research.common.clinical_training import ClinicalTrainingConfig, evaluate_split, make_nrm_v2_optimizer
from research.common.metrics import summarize_epoch
from research.common.resnet import loss_grad_fn
from research.phase0_baseline.run import batch_iterator
from research.common.correctness_memory_io import correctness_state_summary, save_frozen_correctness_artifacts
from research.common.memory import extract_nrm_v2_state, memory_state_summary
from research.common.resnet import init_resnet18_params, resnet18_apply, resnet18_features
from research.common.trajectory_store import MemoryCheckpointStore
import optax


def per_sample_correctness_batch(
    params: dict[str, Any],
    x_batch: jnp.ndarray,
    y_batch: jnp.ndarray,
    *,
    eps: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build (k, c) for a mini-batch using single-sample forwards at fixed params.

    ResNet batch-norm mixes the batch axis when N > 1; deployment uses N = 1.
    """
    batch_size = int(x_batch.shape[0])
    keys: list[jnp.ndarray] = []
    targets: list[jnp.ndarray] = []
    for i in range(batch_size):
        logits_i = resnet18_apply(params, x_batch[i])
        h_i = resnet18_features(params, x_batch[i])
        k_i = normalize_key(h_i, eps=eps)
        c_i = correctness_from_logits(logits_i, y_batch[i])
        keys.append(jnp.asarray(k_i, dtype=jnp.float32).reshape(-1))
        targets.append(jnp.asarray(c_i, dtype=jnp.float32).reshape(()))
    return jnp.stack(keys, axis=0), jnp.stack(targets, axis=0)


def train_one_seed_correctness(
    bundle: ClinicalBundle,
    *,
    seed: int,
    cfg: ClinicalTrainingConfig,
    score_splits: tuple[str, ...] = ("cal", "test"),
) -> tuple[dict, Any, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Train ResNet-18 + NRM v2 + high-dimensional correctness memory (classifier path unchanged)."""
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    params = init_resnet18_params(init_key, num_classes=bundle.num_classes)
    tx = make_nrm_v2_optimizer(cfg)
    opt_state = tx.init(params)
    rng = np.random.default_rng(seed)

    assoc_cfg = cfg.associative_memory
    assoc_mem: CorrectnessMemory | None = None
    checkpoint_store: MemoryCheckpointStore | None = None
    if assoc_cfg.use_associative_memory:
        h_probe = resnet18_features(params, jnp.asarray(bundle.x_train[:1], dtype=jnp.float32))
        d_k = int(h_probe.shape[-1])
        assoc_mem = CorrectnessMemory(d_k, assoc_cfg)
        checkpoint_store = MemoryCheckpointStore.from_config(assoc_cfg)
        for matrix in assoc_mem.state.matrices:
            arr = np.asarray(matrix)
            expected = expected_matrix_shape(d_k, assoc_cfg.correctness_z_dim)
            if arr.shape != expected:
                raise ValueError(f"Expected M shape {expected}, got {arr.shape}")

    per_sample_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    global_step = 0
    t0 = time.perf_counter()

    split_data = {
        "cal": (bundle.x_cal, bundle.y_cal, bundle.sample_ids["cal"]),
        "test": (bundle.x_test, bundle.y_test, bundle.sample_ids["test"]),
    }

    n_batches = (len(bundle.x_train) + cfg.batch_size - 1) // cfg.batch_size
    clinical_log(
        f"  [{bundle.task} seed={seed}] correctness-memory training: {len(bundle.x_train)} samples, "
        f"{cfg.epochs} epochs, ~{n_batches} batches/epoch",
        verbose=cfg.verbose,
    )

    for epoch in range(cfg.epochs):
        epoch_t0 = time.perf_counter()
        train_losses: list[float] = []
        for x_batch, y_batch, _ids in batch_iterator(
            bundle.x_train,
            bundle.y_train,
            bundle.sample_ids["train"],
            cfg.batch_size,
            rng,
            shuffle=True,
        ):
            x_j = jnp.asarray(x_batch, dtype=jnp.float32)
            y_j = jnp.asarray(y_batch, dtype=jnp.int32)
            loss, grads = loss_grad_fn(params, x_j, y_j)
            if not jnp.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch}, step={global_step}")

            if assoc_mem is not None:
                # theta_t: single-sample (h_t, y_hat_t) -> c_t, then batch delta on M_t.
                k_batch, c_batch = per_sample_correctness_batch(
                    params,
                    x_j,
                    y_j,
                    eps=assoc_cfg.eps,
                )
                assoc_mem.update(k_batch, c_batch)

            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            train_losses.append(float(loss))
            global_step += 1

            if checkpoint_store is not None:
                checkpoint_store.save_memory_checkpoint(global_step, assoc_mem.state)

        mean_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        for split_name in score_splits:
            if split_name not in split_data:
                continue
            x_split, y_split, ids = split_data[split_name]
            split_df = evaluate_split(
                params,
                x_split,
                y_split,
                ids,
                opt_state,
                seed=seed,
                epoch=epoch,
                step=global_step,
                split=split_name,
                cfg=cfg,
                score_msa=False,
            )
            per_sample_parts.append(split_df)

        epoch_eval = pd.concat(per_sample_parts[-len(score_splits) :], ignore_index=True)
        epoch_rows: list[dict[str, Any]] = []
        for row in summarize_epoch(epoch_eval, train_loss=mean_train_loss):
            row["task"] = bundle.task
            row["epoch_time_s"] = time.perf_counter() - epoch_t0
            row.update(memory_state_summary(extract_nrm_v2_state(opt_state)))
            if assoc_mem is not None:
                row.update(correctness_state_summary(assoc_mem.state))
            metric_rows.append(row)
            epoch_rows.append(row)

        cal_acc = next((r["accuracy"] for r in epoch_rows if r["split"] == "cal"), float("nan"))
        test_acc = next((r["accuracy"] for r in epoch_rows if r["split"] == "test"), float("nan"))
        clinical_log(
            f"  [{bundle.task} seed={seed}] epoch {epoch + 1}/{cfg.epochs}: "
            f"train_loss={mean_train_loss:.4f} cal_acc={cal_acc:.3f} test_acc={test_acc:.3f} "
            f"({time.perf_counter() - epoch_t0:.1f}s)",
            verbose=cfg.verbose,
        )

    per_sample = pd.concat(per_sample_parts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    last_test = metrics[(metrics["split"] == "test") & (metrics["epoch"] == cfg.epochs - 1)]
    test_acc = float(last_test["accuracy"].iloc[0]) if len(last_test) else float("nan")
    summary = {
        "task": bundle.task,
        "seed": seed,
        "memory_kind": "correctness",
        "optimizer": "nrm_v2_observe",
        "architecture": "resnet18",
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "long_term_depth": cfg.long_term_depth,
        "memory_tau": cfg.memory_tau,
        "attention_tau": cfg.attention_tau,
        "n_train": int(len(bundle.x_train)),
        "n_cal": int(len(bundle.x_cal)),
        "n_test": int(len(bundle.x_test)),
        "test_acc": test_acc,
        "train_time_s": time.perf_counter() - t0,
        "final_step": global_step,
        "correctness_d_k": int(assoc_mem.d_k) if assoc_mem is not None else None,
        "correctness_z_dim": int(assoc_mem.d_v) if assoc_mem is not None else None,
    }

    if assoc_mem is not None and checkpoint_store is not None:
        checkpoint_store.save_final_checkpoint(global_step, assoc_mem.state)
        frozen_checkpoints = checkpoint_store.freeze()
        assoc_summary = save_frozen_correctness_artifacts(
            cfg.output_dir,
            task=bundle.task,
            seed=seed,
            state=assoc_mem.state,
            checkpoints=frozen_checkpoints,
            cfg=assoc_cfg,
        )
        summary["associative_memory"] = True
        summary["correctness_memory"] = True
        summary.update(assoc_summary)
        summary["checkpoint_count"] = int(frozen_checkpoints.T)
        summary["checkpoint_steps"] = frozen_checkpoints.checkpoint_steps.tolist()
    else:
        summary["associative_memory"] = False
        summary["correctness_memory"] = False

    return params, opt_state, per_sample, metrics, summary
