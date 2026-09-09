"""Phase 5 — distribution-shift / OOD detection analysis."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, average_precision_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from research.common.memory_training import (
    ATTENTION_TAU,
    BATCH_SIZE,
    DEFAULT_SEEDS,
    EPOCHS,
    LEARNING_RATE,
    LONG_TERM_DEPTH,
    MEMORY_TAU,
    TRAIN_SUBSET,
    MemoryPhaseConfig,
    evaluate_split_with_memory,
    make_nrm_v2_observe_optimizer,
)
from research.common.model import init_mlp_params, loss_grad_fn
from research.common.shift_datasets import (
    DEFAULT_OOD_SHIFTS,
    ShiftDataset,
    build_shift_dataset,
    list_default_ood_shifts,
    load_id_test_split,
)
from research.phase0_baseline.run import batch_iterator, load_clean_mnist_bundle

OOD_SCORERS: dict[str, str] = {
    "predictive_entropy": "predictive_entropy",
    "memory_novelty": "memory_novelty",
    "memory_disagreement": "memory_disagreement",
}

COMBINED_OOD_FEATURES = list(OOD_SCORERS.values())
DEFAULT_MIN_AUROC = 0.55
DEFAULT_AUROC_MARGIN = 0.02


def ood_label_array(n_id: int, n_ood: int) -> np.ndarray:
    """0 = in-distribution (ID), 1 = out-of-distribution (OOD)."""
    return np.concatenate([np.zeros(n_id, dtype=int), np.ones(n_ood, dtype=int)])


def score_ood_auroc_auprc(y_ood: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y_ood = np.asarray(y_ood, dtype=int)
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(scores)
    y_ood = y_ood[mask]
    scores = scores[mask]
    if len(y_ood) == 0 or len(np.unique(y_ood)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "n": 0}
    return {
        "auroc": float(roc_auc_score(y_ood, scores)),
        "auprc": float(average_precision_score(y_ood, scores)),
        "n": int(len(y_ood)),
    }


def _valid_mask(df: pd.DataFrame, columns: Iterable[str]) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)
    for col in columns:
        mask &= df[col].notna().to_numpy()
    return mask


def evaluate_shift_scorers(
    id_df: pd.DataFrame,
    ood_df: pd.DataFrame,
    *,
    scorers: dict[str, str] | None = None,
) -> pd.DataFrame:
    scorers = scorers or OOD_SCORERS
    rows: list[dict[str, Any]] = []
    shift_name = str(ood_df["shift"].iloc[0]) if len(ood_df) else "unknown"
    for scorer_name, column in scorers.items():
        id_scores = id_df[column].to_numpy(dtype=float)
        ood_scores = ood_df[column].to_numpy(dtype=float)
        scores = np.concatenate([id_scores, ood_scores])
        labels = ood_label_array(len(id_scores), len(ood_scores))
        metrics = score_ood_auroc_auprc(labels, scores)
        rows.append(
            {
                "shift": shift_name,
                "scorer": scorer_name,
                "column": column,
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "n_id": len(id_scores),
                "n_ood": len(ood_scores),
                "n": metrics["n"],
            }
        )
    return pd.DataFrame(rows)


def fit_combined_ood_scorer(
    id_df: pd.DataFrame,
    ood_df: pd.DataFrame,
    feature_columns: list[str],
) -> Pipeline:
    train_df = pd.concat([id_df, ood_df], ignore_index=True)
    labels = np.concatenate(
        [np.zeros(len(id_df), dtype=int), np.ones(len(ood_df), dtype=int)]
    )
    mask = _valid_mask(train_df, feature_columns)
    if not mask.any():
        raise ValueError("No valid rows available to fit combined OOD scorer.")
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=0)),
        ]
    )
    model.fit(train_df.loc[mask, feature_columns].to_numpy(dtype=float), labels[mask])
    return model


def predict_ood_probability(model: Pipeline, df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    mask = _valid_mask(df, feature_columns)
    scores = np.full(len(df), np.nan, dtype=float)
    if not mask.any():
        return scores
    scores[mask] = model.predict_proba(df.loc[mask, feature_columns].to_numpy(dtype=float))[:, 1]
    return scores


def evaluate_combined_ood_models(
    id_train: pd.DataFrame,
    ood_train: pd.DataFrame,
    id_eval: pd.DataFrame,
    ood_eval: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, Pipeline]:
    features = list(feature_columns or COMBINED_OOD_FEATURES)
    model = fit_combined_ood_scorer(id_train, ood_train, features)
    id_scores = predict_ood_probability(model, id_eval, features)
    ood_scores = predict_ood_probability(model, ood_eval, features)
    labels = ood_label_array(len(id_scores), len(ood_scores))
    combined_scores = np.concatenate([id_scores, ood_scores])
    metrics = score_ood_auroc_auprc(labels, combined_scores)
    shift_name = str(ood_eval["shift"].iloc[0]) if len(ood_eval) else "unknown"
    row = pd.DataFrame(
        [
            {
                "shift": shift_name,
                "scorer": "combined",
                "column": "combined_ood_probability",
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "n_id": len(id_scores),
                "n_ood": len(ood_scores),
                "n": metrics["n"],
            }
        ]
    )
    return row, model


def train_and_score_ood_shifts(
    *,
    seed: int,
    cfg: MemoryPhaseConfig,
    shifts: Iterable[str],
    shift_params: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Train on clean MNIST and score ID/OOD splits at the final epoch."""
    bundle = load_clean_mnist_bundle(cfg.data_dir, train_subset=cfg.train_subset)
    shift_params = shift_params or DEFAULT_OOD_SHIFTS
    ood_datasets: dict[str, ShiftDataset] = {}
    for shift_name in shifts:
        ood_datasets[shift_name] = build_shift_dataset(
            shift_name=shift_name,
            x_source=bundle.x_test,
            y_source=bundle.y_test,
            sample_ids=bundle.sample_ids["test"],
            shift_params=shift_params.get(shift_name),
            data_dir=cfg.data_dir,
        )

    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    params = init_mlp_params(init_key)
    tx = make_nrm_v2_observe_optimizer(cfg)
    opt_state = tx.init(params)
    rng = np.random.default_rng(seed)
    global_step = 0

    for epoch in range(cfg.epochs):
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
            global_step += 1

    eval_cfg = MemoryPhaseConfig(
        phase_level=3,
        seeds=cfg.seeds,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        train_subset=cfg.train_subset,
        long_term_depth=cfg.long_term_depth,
        memory_tau=cfg.memory_tau,
        attention_tau=cfg.attention_tau,
        data_dir=cfg.data_dir,
        output_dir=cfg.output_dir,
    )
    parts: list[pd.DataFrame] = []

    id_df = evaluate_split_with_memory(
        params,
        bundle.x_test,
        bundle.y_test,
        bundle.sample_ids["test"],
        opt_state,
        seed=seed,
        epoch=cfg.epochs - 1,
        step=global_step,
        split="test",
        cfg=eval_cfg,
    )
    id_df["domain"] = "id"
    id_df["shift"] = "mnist"
    id_df["is_ood"] = 0
    parts.append(id_df)

    for shift_name, dataset in ood_datasets.items():
        ood_df = evaluate_split_with_memory(
            params,
            dataset.x,
            dataset.y,
            dataset.sample_ids,
            opt_state,
            seed=seed,
            epoch=cfg.epochs - 1,
            step=global_step,
            split="ood",
            cfg=eval_cfg,
        )
        ood_df["domain"] = "ood"
        ood_df["shift"] = shift_name
        ood_df["is_ood"] = 1
        parts.append(ood_df)

    return pd.concat(parts, ignore_index=True)


def collect_ood_per_sample(
    *,
    seeds: tuple[int, ...],
    cfg: MemoryPhaseConfig,
    shifts: Iterable[str],
    shift_params: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed in seeds:
        print(f"=== phase5 ood scoring seed={seed} ===")
        t0 = time.perf_counter()
        frame = train_and_score_ood_shifts(
            seed=seed,
            cfg=cfg,
            shifts=shifts,
            shift_params=shift_params,
        )
        frames.append(frame)
        print(f"  rows={len(frame):,}, time={time.perf_counter() - t0:.1f}s")
    return pd.concat(frames, ignore_index=True)


def plot_roc_curves(
    id_df: pd.DataFrame,
    ood_df: pd.DataFrame,
    out_dir: Path,
    *,
    scorers: dict[str, str] | None = None,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scorers = scorers or OOD_SCORERS
    shift_name = str(ood_df["shift"].iloc[0])
    paths: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(7, 5))
    for scorer_name, column in scorers.items():
        id_scores = id_df[column].to_numpy(dtype=float)
        ood_scores = ood_df[column].to_numpy(dtype=float)
        scores = np.concatenate([id_scores, ood_scores])
        labels = ood_label_array(len(id_scores), len(ood_scores))
        mask = np.isfinite(scores)
        if mask.sum() == 0 or len(np.unique(labels[mask])) < 2:
            continue
        fpr, tpr, _ = roc_curve(labels[mask], scores[mask])
        ax.plot(fpr, tpr, label=f"{scorer_name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC curves — {shift_name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / f"roc_curves_{shift_name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    paths["roc_curves"] = path
    return paths


def plot_id_ood_score_distributions(
    id_df: pd.DataFrame,
    ood_df: pd.DataFrame,
    out_dir: Path,
    *,
    scorers: dict[str, str] | None = None,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scorers = scorers or OOD_SCORERS
    shift_name = str(ood_df["shift"].iloc[0])
    paths: dict[str, Path] = {}

    for scorer_name, column in scorers.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        id_vals = id_df[column].dropna()
        ood_vals = ood_df[column].dropna()
        ax.hist(id_vals, bins=30, alpha=0.65, label="ID (MNIST)", color="steelblue")
        ax.hist(ood_vals, bins=30, alpha=0.65, label=f"OOD ({shift_name})", color="darkorange")
        ax.set_xlabel(column)
        ax.set_ylabel("count")
        ax.set_title(f"ID/OOD score distribution — {scorer_name}")
        ax.legend()
        fig.tight_layout()
        path = out_dir / f"id_ood_distribution_{shift_name}_{scorer_name}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths[scorer_name] = path
    return paths


def plot_uncertainty_histograms(
    id_df: pd.DataFrame,
    ood_df: pd.DataFrame,
    out_dir: Path,
    *,
    scorers: dict[str, str] | None = None,
) -> dict[str, Path]:
    """Population-level uncertainty histograms (ID vs OOD overlaid)."""
    return plot_id_ood_score_distributions(id_df, ood_df, out_dir, scorers=scorers)


def decision_gate(
    metrics_df: pd.DataFrame,
    *,
    min_auroc: float = DEFAULT_MIN_AUROC,
    auroc_margin: float = DEFAULT_AUROC_MARGIN,
) -> dict[str, Any]:
    """PASS when at least one shift/scorer meaningfully separates ID from OOD."""
    valid = metrics_df[np.isfinite(metrics_df["auroc"])]
    if valid.empty:
        return {
            "passed": False,
            "best_shift": "",
            "best_scorer": "",
            "best_auroc": float("nan"),
            "entropy_best_auroc": float("nan"),
            "msa_beats_entropy": False,
            "n_shifts_passing": 0,
            "min_auroc": min_auroc,
            "auroc_margin": auroc_margin,
            "decision": "No valid OOD AUROC results. Do not claim OOD detection.",
            "next_phase": "None — failed Phase 5 decision gate.",
        }

    best_row = valid.loc[valid["auroc"].idxmax()]
    entropy_rows = valid[valid["scorer"] == "predictive_entropy"]
    entropy_best = float(entropy_rows["auroc"].max()) if len(entropy_rows) else float("nan")
    best_auroc = float(best_row["auroc"])
    passing = valid[valid["auroc"] >= min_auroc]
    msa_rows = valid[valid["scorer"].isin(["memory_novelty", "memory_disagreement", "combined"])]
    msa_best = float(msa_rows["auroc"].max()) if len(msa_rows) else float("nan")
    msa_beats_entropy = bool(np.isfinite(msa_best) and np.isfinite(entropy_best) and msa_best > entropy_best + auroc_margin)
    passed = len(passing) > 0

    if passed:
        decision = (
            "At least one uncertainty signal separates ID from OOD on controlled MNIST shifts. "
            "Proceed to Phase 6 (temporal uncertainty)."
        )
        next_phase = "Phase 6 — track temporal uncertainty trajectories per example."
    else:
        decision = (
            "Signals do not reliably separate ID from OOD. "
            "Do not call this OOD detection; stop or revisit signal design."
        )
        next_phase = "None — failed Phase 5 decision gate."

    return {
        "passed": passed,
        "best_shift": str(best_row["shift"]),
        "best_scorer": str(best_row["scorer"]),
        "best_auroc": best_auroc,
        "entropy_best_auroc": entropy_best,
        "msa_best_auroc": msa_best,
        "msa_beats_entropy": msa_beats_entropy,
        "n_shifts_passing": int(passing["shift"].nunique()),
        "min_auroc": min_auroc,
        "auroc_margin": auroc_margin,
        "decision": decision,
        "next_phase": next_phase,
    }


def write_phase5_readme(path: Path, gate: dict[str, Any], metrics_df: pd.DataFrame) -> None:
    pivot = metrics_df.pivot(index="scorer", columns="shift", values="auroc").round(4)
    lines = [
        "# Phase 5 — Distribution-shift test",
        "",
        "## Hypothesis",
        "",
        "Memory-derived MSA signals (novelty, disagreement) and predictive entropy",
        "separate in-distribution MNIST from controlled distribution shifts.",
        "",
        "## Experiment",
        "",
        "- ID: MNIST test",
        "- OOD shifts: rotated, translated, corrupted MNIST, Fashion-MNIST",
        "- Scorers: predictive entropy, MSA novelty, MSA disagreement, combined logistic score",
        "- Metrics: AUROC, AUPRC (label 1 = OOD)",
        "",
        "## Result",
        "",
        f"- Best AUROC: {gate['best_auroc']:.4f} ({gate['best_scorer']} on {gate['best_shift']})",
        f"- Best entropy AUROC: {gate['entropy_best_auroc']:.4f}",
        f"- Shifts passing threshold ({gate['min_auroc']}): {gate['n_shifts_passing']}",
        f"- Decision gate: **{'PASS' if gate['passed'] else 'FAIL'}**",
        "",
        "AUROC by shift:",
        "",
        pivot.to_string(),
        "",
        "## Decision",
        "",
        gate["decision"],
        "",
        "## Next phase",
        "",
        gate["next_phase"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_phase5_analysis(
    per_sample_df: pd.DataFrame,
    output_dir: Path,
    *,
    id_shift_name: str = "mnist",
    calibrate_fraction: float = 0.5,
    scorers: dict[str, str] | None = None,
    combined_features: list[str] | None = None,
    min_auroc: float = DEFAULT_MIN_AUROC,
    auroc_margin: float = DEFAULT_AUROC_MARGIN,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    scorers = scorers or OOD_SCORERS
    combined_cols = list(combined_features or COMBINED_OOD_FEATURES)

    if save_artifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[pd.DataFrame] = []
    ood_shifts = sorted(s for s in per_sample_df["shift"].unique() if s != id_shift_name)

    for shift_name in ood_shifts:
        id_df = per_sample_df[per_sample_df["shift"] == id_shift_name].copy()
        ood_df = per_sample_df[per_sample_df["shift"] == shift_name].copy()
        if id_df.empty or ood_df.empty:
            continue

        # Calibrate combined scorer on a subset of ID/OOD rows (same seed strata).
        id_cal = id_df.groupby("seed", group_keys=False).apply(
            lambda g: g.sample(frac=calibrate_fraction, random_state=0) if len(g) > 1 else g
        )
        ood_cal = ood_df.groupby("seed", group_keys=False).apply(
            lambda g: g.sample(frac=calibrate_fraction, random_state=0) if len(g) > 1 else g
        )
        id_eval = id_df.drop(id_cal.index, errors="ignore")
        ood_eval = ood_df.drop(ood_cal.index, errors="ignore")
        if id_eval.empty:
            id_eval = id_df
        if ood_eval.empty:
            ood_eval = ood_df

        single_metrics = evaluate_shift_scorers(id_eval, ood_eval, scorers=scorers)
        combined_metrics, _model = evaluate_combined_ood_models(
            id_cal,
            ood_cal,
            id_eval,
            ood_eval,
            feature_columns=combined_cols,
        )
        shift_metrics = pd.concat([single_metrics, combined_metrics], ignore_index=True)
        metric_rows.append(shift_metrics)

        if save_artifacts:
            plot_roc_curves(id_eval, ood_eval, plots_dir, scorers=scorers)
            plot_id_ood_score_distributions(id_eval, ood_eval, plots_dir, scorers=scorers)
            plot_uncertainty_histograms(id_eval, ood_eval, plots_dir, scorers=scorers)

    metrics_df = pd.concat(metric_rows, ignore_index=True) if metric_rows else pd.DataFrame()
    gate = decision_gate(metrics_df, min_auroc=min_auroc, auroc_margin=auroc_margin)

    if save_artifacts:
        per_sample_df.to_csv(output_dir / "per_sample.csv", index=False)
        metrics_df.to_csv(output_dir / "metrics.csv", index=False)
        pd.DataFrame([gate]).to_csv(output_dir / "decision_gate.csv", index=False)
        config = {
            "phase": 5,
            "id_shift": id_shift_name,
            "ood_shifts": ood_shifts,
            "scorers": scorers,
            "combined_features": combined_cols,
            "calibrate_fraction": calibrate_fraction,
            "min_auroc": min_auroc,
            "auroc_margin": auroc_margin,
            "decision_gate": gate,
        }
        (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        write_phase5_readme(output_dir / "README.md", gate, metrics_df)

    return {
        "dataframe": per_sample_df,
        "metrics": metrics_df,
        "decision_gate": gate,
        "plots_dir": plots_dir,
        "output_dir": output_dir,
    }
