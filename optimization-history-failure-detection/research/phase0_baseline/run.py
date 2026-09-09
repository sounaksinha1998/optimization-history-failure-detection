"""Phase 0 — Adam baseline instrumentation (no optimizer-update changes)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.common.datasets import (
    NoisyMNISTBundle,
    NoisyMNISTConfig,
    load_mnist_numpy,
    make_sample_ids,
    select_train_subset,
    split_mnist,
)
from research.common.metrics import PHASE0_COLUMNS, records_from_logits, summarize_epoch
from research.common.model import adam_optimizer, init_mlp_params, loss_grad_fn, mlp_apply
from research.common.plotting import plot_phase0

EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
DEFAULT_SEEDS = (42, 123, 456)
TRAIN_SUBSET = 10_000


@dataclass(frozen=True)
class Phase0Config:
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    variant: str = "mnist_clean"
    train_subset: int | None = TRAIN_SUBSET
    data_dir: Path = ROOT / "data"
    output_dir: Path = ROOT / "research" / "phase0_baseline"


def batch_iterator(
    x: np.ndarray,
    y: np.ndarray,
    sample_ids: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    *,
    shuffle: bool,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    indices = np.arange(len(x))
    if shuffle:
        rng.shuffle(indices)
    batches = []
    for start in range(0, len(indices), batch_size):
        idx = indices[start : start + batch_size]
        batches.append((x[idx], y[idx], sample_ids[idx]))
    return batches


def evaluate_split(
    params: dict,
    x: np.ndarray,
    y: np.ndarray,
    sample_ids: np.ndarray,
    *,
    seed: int,
    epoch: int,
    step: int,
    split: str,
) -> pd.DataFrame:
    x_j = jnp.asarray(x, dtype=jnp.float32)
    logits = np.asarray(mlp_apply(params, x_j))
    return records_from_logits(
        logits,
        y,
        sample_ids,
        seed=seed,
        epoch=epoch,
        step=step,
        split=split,
    )


def train_one_seed(
    bundle: NoisyMNISTBundle,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    params = init_mlp_params(init_key)
    tx = adam_optimizer(learning_rate)
    opt_state = tx.init(params)
    rng = np.random.default_rng(seed)

    per_sample_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    global_step = 0
    t0 = time.perf_counter()

    for epoch in range(epochs):
        epoch_t0 = time.perf_counter()
        train_losses: list[float] = []
        for x_batch, y_batch, _ids in batch_iterator(
            bundle.x_train,
            bundle.y_train,
            bundle.sample_ids["train"],
            batch_size,
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
        val_df = evaluate_split(
            params,
            bundle.x_val,
            bundle.y_val,
            bundle.sample_ids["val"],
            seed=seed,
            epoch=epoch,
            step=global_step,
            split="val",
        )
        test_df = evaluate_split(
            params,
            bundle.x_test,
            bundle.y_test,
            bundle.sample_ids["test"],
            seed=seed,
            epoch=epoch,
            step=global_step,
            split="test",
        )
        per_sample_parts.extend([val_df, test_df])
        epoch_eval = pd.concat([val_df, test_df], ignore_index=True)
        for row in summarize_epoch(epoch_eval, train_loss=mean_train_loss):
            row["epoch_time_s"] = time.perf_counter() - epoch_t0
            metric_rows.append(row)

    per_sample = pd.concat(per_sample_parts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    last_test = metrics[(metrics["split"] == "test") & (metrics["epoch"] == epochs - 1)].iloc[0]
    summary = {
        "seed": seed,
        "optimizer": "adam",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "n_train": int(len(bundle.x_train)),
        "test_acc": float(last_test["accuracy"]),
        "test_loss": float(last_test["mean_loss"]),
        "train_time_s": time.perf_counter() - t0,
        "final_step": global_step,
    }
    return per_sample, metrics, summary


def write_phase0_readme(path: Path, cfg: Phase0Config, summaries: list[dict[str, Any]], metrics: pd.DataFrame) -> None:
    val = metrics[metrics["split"] == "val"]
    last_epoch = int(val["epoch"].max()) if len(val) else -1
    last = val[val["epoch"] == last_epoch] if last_epoch >= 0 else val
    mean_acc = float(last["accuracy"].mean()) if len(last) else float("nan")
    mean_loss = float(last["mean_loss"].mean()) if len(last) else float("nan")
    seeds = ", ".join(str(s) for s in cfg.seeds)
    lines = [
        "# Phase 0 — Baseline instrumentation",
        "",
        "## Hypothesis",
        "",
        "An Adam classifier can be instrumented so that every evaluation example yields",
        "prediction, cross-entropy loss, confidence, and predictive entropy without",
        "changing the optimizer update.",
        "",
        "## Experiment",
        "",
        f"- Dataset: MNIST (`{cfg.variant}`)",
        f"- Optimizer: Adam (lr={cfg.learning_rate})",
        f"- Seeds: {seeds}",
        f"- Epochs: {cfg.epochs}, batch size: {cfg.batch_size}, train subset: {cfg.train_subset}",
        "- Logged fields: epoch, step, sample_id, loss, prediction, correctness,",
        "  confidence, predictive entropy",
        "",
        "## Result",
        "",
        f"- Runs completed: {len(summaries)}",
        f"- Mean val accuracy (last epoch): {mean_acc:.4f}",
        f"- Mean val loss (last epoch): {mean_loss:.4f}",
        "- Artifacts: `phase0_baseline.csv`, `per_sample.csv`, `metrics.csv`, `plots/`",
        "",
        "## Decision",
        "",
        "Phase 0 instrumentation is in place. Proceed to Phase 1 (V5B memory as",
        "observation only); do not modify Adam's update.",
        "",
        "## Next phase",
        "",
        "Phase 1 — maintain multi-timescale memories without changing the Adam step;",
        "write `phase1_memory.csv`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_phase0(bundle: NoisyMNISTBundle, cfg: Phase0Config) -> dict[str, Any]:
    output_dir = Path(cfg.output_dir)
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_sample: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    for seed in cfg.seeds:
        print(f"=== phase0 adam seed={seed} ===")
        per_sample, metrics, summary = train_one_seed(
            bundle,
            seed=seed,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
        )
        all_sample.append(per_sample)
        all_metrics.append(metrics)
        summaries.append(summary)
        print(f"  test_acc={summary['test_acc']:.4f}, time={summary['train_time_s']:.1f}s")

    per_sample_df = pd.concat(all_sample, ignore_index=True)[PHASE0_COLUMNS]
    metrics_df = pd.concat(all_metrics, ignore_index=True)

    per_sample_path = output_dir / "per_sample.csv"
    baseline_path = output_dir / "phase0_baseline.csv"
    metrics_path = output_dir / "metrics.csv"
    per_sample_df.to_csv(per_sample_path, index=False)
    per_sample_df.to_csv(baseline_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)

    plot_phase0(metrics_df, per_sample_df, plots_dir)

    config = {
        "seeds": list(cfg.seeds),
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "variant": cfg.variant,
        "train_subset": cfg.train_subset,
        "data_dir": str(cfg.data_dir),
        "output_dir": str(cfg.output_dir),
        "optimizer": "adam",
        "runs": summaries,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_phase0_readme(output_dir / "README.md", cfg, summaries, metrics_df)
    return config


def load_clean_mnist_bundle(
    data_dir: Path,
    *,
    train_subset: int | None = TRAIN_SUBSET,
    split_seed: int = 42,
    val_frac: float = 0.1,
) -> NoisyMNISTBundle:
    """Official MNIST train/val/test with stable sample IDs (no noise)."""
    x_train_full, y_train_full, x_test, y_test = load_mnist_numpy(Path(data_dir))
    x_train, y_train, train_mnist_idx, x_val, y_val, val_mnist_idx = split_mnist(
        x_train_full,
        y_train_full,
        split_seed=split_seed,
        val_frac=val_frac,
    )
    test_mnist_idx = np.arange(len(x_test), dtype=np.int32)
    train_ids = make_sample_ids("train", len(x_train))
    val_ids = make_sample_ids("val", len(x_val))
    test_ids = make_sample_ids("test", len(x_test))
    x_train, y_train, train_ids, train_mnist_idx = select_train_subset(
        x_train,
        y_train,
        train_ids,
        train_mnist_idx,
        train_subset,
        subset_indices_seed=42,
    )
    return NoisyMNISTBundle(
        name="mnist_clean",
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        sample_ids={"train": train_ids, "val": val_ids, "test": test_ids},
        metadata=pd.DataFrame(),
        config=NoisyMNISTConfig(variant="mnist_clean", data_dir=Path(data_dir)),
        sigma=0.0,
        mnist_indices={
            "train": train_mnist_idx,
            "val": val_mnist_idx,
            "test": test_mnist_idx,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 Adam baseline instrumentation.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "research" / "phase0_baseline")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--variant", type=str, default="mnist_clean")
    parser.add_argument("--train-subset", type=int, default=TRAIN_SUBSET)
    args = parser.parse_args()

    cfg = Phase0Config(
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        variant=args.variant,
        train_subset=args.train_subset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    bundle = load_clean_mnist_bundle(cfg.data_dir, train_subset=cfg.train_subset)
    run_phase0(bundle, cfg)
    print(f"Wrote {cfg.output_dir / 'phase0_baseline.csv'}")


if __name__ == "__main__":
    main()
