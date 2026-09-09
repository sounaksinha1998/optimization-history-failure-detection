"""ResNet-18 training with frozen NRM v2 optimization-history memory."""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

from optimizer import research_nrm_v2_observe
from optimizer.associative_memory import (
    AssociativeMemory,
    AssociativeMemoryConfig,
    normalize_key,
    value_from_logits,
)
from research.common.clinical_datasets import ClinicalBundle, ClinicalTask
from research.common.clinical_log import clinical_log
from research.common.memory import (
    associative_state_summary,
    extract_nrm_v2_state,
    memory_state_summary,
    save_frozen_associative_artifacts,
)
from research.common.metrics import records_from_logits, summarize_epoch
from research.common.msa import per_sample_msa_signals
from research.common.resnet import (
    example_grad,
    init_resnet18_params,
    loss_grad_fn,
    resnet18_apply,
    resnet18_features,
)
from research.common.trajectory_store import MemoryCheckpointStore
from research.phase0_baseline.run import batch_iterator

DEFAULT_SEEDS = (42, 123, 456)
EPOCHS = 8
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
LONG_TERM_DEPTH = 3
MEMORY_TAU = 8.0
ATTENTION_TAU = 1.0


@dataclass(frozen=True)
class ClinicalTrainingConfig:
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    long_term_depth: int = LONG_TERM_DEPTH
    memory_tau: float = MEMORY_TAU
    attention_tau: float = ATTENTION_TAU
    output_dir: Path = Path("research/final_experiment")
    verbose: int = 1
    associative_memory: AssociativeMemoryConfig = field(default_factory=AssociativeMemoryConfig)


def make_nrm_v2_optimizer(cfg: ClinicalTrainingConfig) -> optax.GradientTransformation:
    return research_nrm_v2_observe(
        learning_rate=cfg.learning_rate,
        tau=cfg.memory_tau,
        long_term_depth=cfg.long_term_depth,
    )


def checkpoint_path(output_dir: Path, task: ClinicalTask, seed: int) -> Path:
    return output_dir / "checkpoints" / f"{task}_seed{seed}.pkl"


def memory_path(output_dir: Path, task: ClinicalTask, seed: int) -> Path:
    return output_dir / "memory" / f"{task}_seed{seed}_memory.npz"


def save_checkpoint(
    path: Path,
    *,
    params: dict,
    opt_state: Any,
    task: ClinicalTask,
    seed: int,
    cfg: ClinicalTrainingConfig,
    summary: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": params,
        "opt_state": opt_state,
        "task": task,
        "seed": seed,
        "config": asdict(cfg),
        "summary": summary,
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def load_checkpoint(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def save_frozen_memory(path: Path, opt_state: Any, *, task: ClinicalTask, seed: int) -> dict[str, float]:
    mem_state = extract_nrm_v2_state(opt_state)
    summary = memory_state_summary(mem_state)
    path.parent.mkdir(parents=True, exist_ok=True)

    long_term_arrays = []
    for level in mem_state.long_term:
        leaves = jax.tree_util.tree_leaves(level)
        long_term_arrays.append(np.concatenate([np.asarray(x).reshape(-1) for x in leaves]))

    np.savez_compressed(
        path,
        task=task,
        seed=seed,
        optimizer_step=np.array(summary["optimizer_step"]),
        long_term=np.array(long_term_arrays, dtype=object),
        summary_json=np.array(json.dumps(summary)),
    )
    return {k: float(v) if isinstance(v, (int, float)) else v for k, v in summary.items()}


def load_frozen_memory_state(checkpoint_payload: dict[str, Any]) -> Any:
    """Return optimizer state with frozen memory from checkpoint."""
    return checkpoint_payload["opt_state"]


def evaluate_split(
    params: dict,
    x: np.ndarray,
    y: np.ndarray,
    sample_ids: np.ndarray,
    opt_state: Any,
    *,
    seed: int,
    epoch: int,
    step: int,
    split: str,
    cfg: ClinicalTrainingConfig,
    score_msa: bool = False,
) -> pd.DataFrame:
    mem_state = extract_nrm_v2_state(opt_state)
    mem_summary = memory_state_summary(mem_state)

    logits_list: list[np.ndarray] = []
    batch_size = cfg.batch_size
    for start in range(0, len(x), batch_size):
        xb = jnp.asarray(x[start : start + batch_size], dtype=jnp.float32)
        logits_list.append(np.asarray(resnet18_apply(params, xb)))
    logits = np.concatenate(logits_list, axis=0)

    base_df = records_from_logits(
        logits,
        y,
        sample_ids,
        seed=seed,
        epoch=epoch,
        step=step,
        split=split,
    )
    for key, value in mem_summary.items():
        base_df[key] = value

    if not score_msa:
        return base_df

    signal_rows: list[dict[str, float]] = []
    grad_norms: list[float] = []
    n_samples = len(x)
    log_every = max(1, n_samples // 10)
    for i in range(n_samples):
        if cfg.verbose >= 2 and (i == 0 or (i + 1) % log_every == 0 or i + 1 == n_samples):
            clinical_log(
                f"    MSA scoring {split}: {i + 1}/{n_samples}",
                verbose=cfg.verbose,
                level=2,
            )
        xi = jnp.asarray(x[i], dtype=jnp.float32)
        yi = jnp.asarray(int(y[i]), dtype=jnp.int32)
        grad_i = example_grad(params, xi, yi)
        leaves = jax.tree_util.tree_leaves(grad_i)
        grad_norm = float(jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)))
        grad_norms.append(grad_norm)
        signal_rows.append(
            per_sample_msa_signals(
                grad_i,
                mem_state.long_term,
                tau=cfg.attention_tau,
            )
        )
    signal_df = pd.DataFrame(signal_rows)
    signal_df["gradient_norm"] = grad_norms
    return pd.concat([base_df.reset_index(drop=True), signal_df.reset_index(drop=True)], axis=1)


def train_one_seed(
    bundle: ClinicalBundle,
    *,
    seed: int,
    cfg: ClinicalTrainingConfig,
    score_splits: tuple[str, ...] = ("cal", "test"),
) -> tuple[dict, Any, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    params = init_resnet18_params(init_key, num_classes=bundle.num_classes)
    tx = make_nrm_v2_optimizer(cfg)
    opt_state = tx.init(params)
    rng = np.random.default_rng(seed)

    assoc_cfg = cfg.associative_memory
    assoc_mem: AssociativeMemory | None = None
    checkpoint_store: MemoryCheckpointStore | None = None
    if assoc_cfg.use_associative_memory:
        h_probe = resnet18_features(params, jnp.asarray(bundle.x_train[:1], dtype=jnp.float32))
        d_k = int(h_probe.shape[-1])
        d_v = int(bundle.num_classes)
        assoc_mem = AssociativeMemory(d_k, d_v, assoc_cfg)
        checkpoint_store = MemoryCheckpointStore.from_config(assoc_cfg)

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
        f"  [{bundle.task} seed={seed}] training: {len(bundle.x_train)} samples, "
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
            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            train_losses.append(float(loss))
            global_step += 1

            if assoc_mem is not None and checkpoint_store is not None:
                # Training-time construction: moving k_t, v_t update M. Do not
                # record R from h_t — evaluation replays frozen h_T after training.
                logits = resnet18_apply(params, x_j)
                h_batch = resnet18_features(params, x_j)
                k_batch = normalize_key(h_batch, eps=assoc_cfg.eps)
                v_batch = value_from_logits(logits, y_j, bundle.num_classes)
                assoc_mem.update(k_batch, v_batch)
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
                row.update(associative_state_summary(assoc_mem.state))
            metric_rows.append(row)
            epoch_rows.append(row)

        cal_acc = next((r["accuracy"] for r in epoch_rows if r["split"] == "cal"), float("nan"))
        test_acc = next((r["accuracy"] for r in epoch_rows if r["split"] == "test"), float("nan"))
        epoch_time_s = time.perf_counter() - epoch_t0
        clinical_log(
            f"  [{bundle.task} seed={seed}] epoch {epoch + 1}/{cfg.epochs}: "
            f"train_loss={mean_train_loss:.4f} cal_acc={cal_acc:.3f} test_acc={test_acc:.3f} "
            f"({epoch_time_s:.1f}s)",
            verbose=cfg.verbose,
        )

    per_sample = pd.concat(per_sample_parts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    last_test = metrics[(metrics["split"] == "test") & (metrics["epoch"] == cfg.epochs - 1)]
    test_acc = float(last_test["accuracy"].iloc[0]) if len(last_test) else float("nan")
    summary = {
        "task": bundle.task,
        "seed": seed,
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
    }

    if assoc_mem is not None and checkpoint_store is not None:
        checkpoint_store.save_final_checkpoint(global_step, assoc_mem.state)
        frozen_checkpoints = checkpoint_store.freeze()
        assoc_summary = save_frozen_associative_artifacts(
            cfg.output_dir,
            task=bundle.task,
            seed=seed,
            state=assoc_mem.state,
            checkpoints=frozen_checkpoints,
            cfg=assoc_cfg,
        )
        summary["associative_memory"] = True
        summary.update(assoc_summary)
        summary["checkpoint_count"] = int(frozen_checkpoints.T)
        summary["checkpoint_steps"] = frozen_checkpoints.checkpoint_steps.tolist()
    else:
        summary["associative_memory"] = False

    return params, opt_state, per_sample, metrics, summary


def export_scoring_artifacts_for_nro(
    bundle: ClinicalBundle,
    *,
    params: dict,
    opt_state: Any,
    seed: int,
    cfg: ClinicalTrainingConfig,
    output_dir: Path,
) -> None:
    """Write ``calibration/`` and ``scored/`` CSVs for :mod:`nro_final_experiment`."""
    from research.common.clinical_failure import build_all_population_frames

    cal_dir = Path(output_dir) / "calibration"
    scored_dir = Path(output_dir) / "scored"
    cal_dir.mkdir(parents=True, exist_ok=True)
    scored_dir.mkdir(parents=True, exist_ok=True)

    cal_df = evaluate_split(
        params,
        bundle.x_cal,
        bundle.y_cal,
        bundle.sample_ids["cal"],
        opt_state,
        seed=seed,
        epoch=cfg.epochs - 1,
        step=-1,
        split="calibration",
        cfg=cfg,
        score_msa=True,
    )
    cal_df["task"] = bundle.task
    cal_df["dataset"] = bundle.task
    cal_df["domain"] = "calibration"
    cal_df.to_csv(cal_dir / f"{bundle.task}_seed{seed}.csv", index=False)

    scored_df = build_all_population_frames(
        bundle,
        task=bundle.task,
        seed=seed,
        train_cfg=cfg,
        checkpoint_path=checkpoint_path(output_dir, bundle.task, seed),
        corruption_base_seed=seed + 10_000,
        verbose=0,
    )
    scored_df.to_csv(scored_dir / f"{bundle.task}_seed{seed}.csv", index=False)


def build_model_manifest(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "architecture": "resnet18",
        "optimizer": "nrm_v2_observe",
        "runs": summaries,
    }
