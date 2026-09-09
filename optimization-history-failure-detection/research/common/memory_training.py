"""Shared V5B-observe training loop for research phases 1–3."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

from optimizer import research_nrm_v2_observe
from research.common.memory import extract_nrm_v2_state, memory_state_summary
from research.common.metrics import (
    PHASE3_SIGNAL_COLUMNS,
    records_from_logits,
    summarize_epoch,
)
from research.common.model import init_mlp_params, loss_grad_fn, mlp_apply
from research.common.msa import (
    attention_from_alignments,
    example_grad,
    gradient_memory_alignments,
    per_sample_msa_signals,
)
from research.phase0_baseline.run import batch_iterator

PhaseLevel = Literal[1, 2, 3]

DEFAULT_SEEDS = (42, 123, 456)
EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
TRAIN_SUBSET = 10_000
LONG_TERM_DEPTH = 3
MEMORY_TAU = 8.0
ATTENTION_TAU = 1.0


@dataclass(frozen=True)
class MemoryPhaseConfig:
    phase_level: PhaseLevel
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    variant: str = "mnist_clean"
    train_subset: int | None = TRAIN_SUBSET
    long_term_depth: int = LONG_TERM_DEPTH
    memory_tau: float = MEMORY_TAU
    attention_tau: float = ATTENTION_TAU
    data_dir: Path = Path(__file__).resolve().parents[2] / "data"
    output_dir: Path = Path(__file__).resolve().parents[1] / "phase1_memory"


def make_nrm_v2_observe_optimizer(cfg: MemoryPhaseConfig) -> optax.GradientTransformation:
    return research_nrm_v2_observe(
        learning_rate=cfg.learning_rate,
        tau=cfg.memory_tau,
        long_term_depth=cfg.long_term_depth,
    )


def evaluate_split_with_memory(
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
    cfg: MemoryPhaseConfig,
) -> pd.DataFrame:
    """Evaluate a split and attach memory / MSA diagnostics per example."""
    x_j = jnp.asarray(x, dtype=jnp.float32)
    logits = np.asarray(mlp_apply(params, x_j))
    base_df = records_from_logits(
        logits,
        y,
        sample_ids,
        seed=seed,
        epoch=epoch,
        step=step,
        split=split,
    )

    mem_state = extract_nrm_v2_state(opt_state)
    mem_summary = memory_state_summary(mem_state)
    for key, value in mem_summary.items():
        base_df[key] = value

    # Pad missing long-term norm columns when depth < 3.
    for j in range(1, cfg.long_term_depth + 1):
        col = f"mem_L{j}_norm"
        if col not in base_df.columns:
            base_df[col] = float("nan")
    for j in range(cfg.long_term_depth + 1, 4):
        base_df[f"mem_L{j}_norm"] = float("nan")

    if cfg.phase_level < 2:
        return base_df

    signal_rows: list[dict[str, float]] = []
    for i in range(len(x)):
        xi = jnp.asarray(x[i], dtype=jnp.float32)
        yi = jnp.asarray(int(y[i]), dtype=jnp.int32)
        grad_i = example_grad(params, xi, yi)
        if cfg.phase_level >= 3:
            signal_rows.append(
                per_sample_msa_signals(
                    grad_i,
                    mem_state.long_term,
                    tau=cfg.attention_tau,
                )
            )
        else:
            alignments = gradient_memory_alignments(grad_i, mem_state.long_term)
            alpha = attention_from_alignments(alignments, tau=cfg.attention_tau)
            row: dict[str, float] = {}
            for j, c_j in enumerate(alignments, start=1):
                row[f"c_{j}"] = float(c_j)
            for j, a_j in enumerate(alpha, start=1):
                row[f"alpha_{j}"] = float(a_j)
            for key in PHASE3_SIGNAL_COLUMNS:
                row[key] = float("nan")
            signal_rows.append(row)

    signal_df = pd.DataFrame(signal_rows)
    for j in range(cfg.long_term_depth + 1, 4):
        signal_df[f"c_{j}"] = float("nan")
        signal_df[f"alpha_{j}"] = float("nan")

    return pd.concat([base_df.reset_index(drop=True), signal_df.reset_index(drop=True)], axis=1)


def train_one_seed(
    bundle,
    *,
    seed: int,
    cfg: MemoryPhaseConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    params = init_mlp_params(init_key)
    tx = make_nrm_v2_observe_optimizer(cfg)
    opt_state = tx.init(params)
    rng = np.random.default_rng(seed)

    per_sample_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    global_step = 0
    t0 = time.perf_counter()

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

        mean_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        for split_name, x_split, y_split, ids in (
            ("val", bundle.x_val, bundle.y_val, bundle.sample_ids["val"]),
            ("test", bundle.x_test, bundle.y_test, bundle.sample_ids["test"]),
        ):
            split_df = evaluate_split_with_memory(
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
            )
            per_sample_parts.append(split_df)

        epoch_eval = pd.concat(per_sample_parts[-2:], ignore_index=True)
        for row in summarize_epoch(epoch_eval, train_loss=mean_train_loss):
            row["epoch_time_s"] = time.perf_counter() - epoch_t0
            mem_state = extract_nrm_v2_state(opt_state)
            row.update(memory_state_summary(mem_state))
            metric_rows.append(row)

    per_sample = pd.concat(per_sample_parts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    last_test = metrics[(metrics["split"] == "test") & (metrics["epoch"] == cfg.epochs - 1)].iloc[0]
    summary = {
        "seed": seed,
        "optimizer": "nrm_v2_observe",
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "long_term_depth": cfg.long_term_depth,
        "memory_tau": cfg.memory_tau,
        "attention_tau": cfg.attention_tau,
        "n_train": int(len(bundle.x_train)),
        "test_acc": float(last_test["accuracy"]),
        "test_loss": float(last_test["mean_loss"]),
        "train_time_s": time.perf_counter() - t0,
        "final_step": global_step,
    }
    return per_sample, metrics, summary
