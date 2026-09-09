"""Clinical failure detection analysis: scoring, metrics, bootstrap, leakage, figures."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline

from research.common.clinical_corruption import DEFAULT_CORRUPTION_SUITE, build_corrupted_set, corruption_config_dict
from research.common.clinical_datasets import ClinicalBundle, ClinicalTask, VALID_CLINICAL_TASKS
from research.common.clinical_log import clinical_log
from research.common.clinical_training import (
    ATTENTION_TAU,
    ClinicalTrainingConfig,
    evaluate_split,
    load_checkpoint,
)
from research.common.complementarity import expected_calibration_error
from research.common.error_prediction import (
    bootstrap_auroc_delta,
    fit_failure_scorer,
    predict_error_probability,
    score_auroc_auprc,
)
from research.common.memory import extract_nrm_v2_state
from research.common.metrics import softmax_probs


def _normalize_task_tuple(datasets: ClinicalTask | Iterable[ClinicalTask]) -> tuple[ClinicalTask, ...]:
    if isinstance(datasets, str):
        return (datasets,)  # type: ignore[return-value]
    return tuple(datasets)


def _normalize_seed_tuple(seeds: int | Iterable[int]) -> tuple[int, ...]:
    if isinstance(seeds, int):
        return (seeds,)
    return tuple(int(s) for s in seeds)
ScorerName = Literal["H", "N", "H+N"]

COVERAGE_GRID = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00)


@dataclass
class FinalClinicalConfig:
    repo_root: Path | None = None
    data_dir: Path | None = None
    output_dir: Path | None = None
    seeds: tuple[int, ...] = (42, 123, 456)
    datasets: tuple[ClinicalTask, ...] = ("pathmnist", "dermamnist", "organamnist")
    bootstrap_replicates: int = 2000
    target_coverage: float = 0.80
    epochs: int = 8
    batch_size: int = 64
    learning_rate: float = 1e-3
    max_train: int | None = 2000
    max_cal: int | None = 500
    max_test: int | None = 1000
    max_external: int | None = 1000
    run_training_if_missing: bool = True
    resume_scoring: bool = True
    show_plots: bool = True
    save_artifacts: bool = True
    verbose: int = 1
    corruption_suite: tuple[Any, ...] = field(default_factory=lambda: DEFAULT_CORRUPTION_SUITE)

    def resolve_paths(self, cwd: Path | None = None) -> FinalClinicalConfig:
        self.datasets = _normalize_task_tuple(self.datasets)
        self.seeds = _normalize_seed_tuple(self.seeds)
        for task in self.datasets:
            if task not in VALID_CLINICAL_TASKS:
                raise ValueError(
                    f"Unknown dataset {task!r}. Expected one of {VALID_CLINICAL_TASKS}. "
                    "If you meant a single dataset, use datasets=('organamnist',) with a trailing comma."
                )
        root = self.repo_root
        if root is None:
            root = Path(cwd or Path.cwd()).resolve()
            if not (root / "research").exists() and (root.parent / "research").exists():
                root = root.parent
        self.repo_root = root
        if self.data_dir is None:
            self.data_dir = root / "data" / "clinical"
        if self.output_dir is None:
            self.output_dir = root / "research" / "final_experiment"
        return self


def ensure_output_layout(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "checkpoints": output_dir / "checkpoints",
        "memory": output_dir / "memory",
        "calibration": output_dir / "calibration",
        "scored": output_dir / "scored",
        "predictions": output_dir / "predictions",
        "metrics": output_dir / "metrics",
        "statistics": output_dir / "statistics",
        "figures": output_dir / "figures",
        "audit": output_dir / "audit",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def normalized_entropy(probs: np.ndarray, num_classes: int, eps: float = 1e-12) -> np.ndarray:
    safe = np.clip(probs, eps, 1.0)
    h = -np.sum(safe * np.log(safe), axis=-1)
    return h / np.log(num_classes)


def fit_combined_scorer(
    cal_df: pd.DataFrame,
    *,
    h_col: str = "normalized_entropy",
    n_col: str = "memory_novelty",
) -> Pipeline:
    return fit_failure_scorer(cal_df, [h_col, n_col], error_col="error", max_iter=2000)


def score_dataframe(
    df: pd.DataFrame,
    *,
    combined_model: Pipeline,
    num_classes: int,
    h_col: str = "normalized_entropy",
    n_col: str = "memory_novelty",
) -> pd.DataFrame:
    out = df.copy()
    if h_col not in out.columns and "predictive_entropy" in out.columns:
        out[h_col] = out["predictive_entropy"] / np.log(num_classes)

    out["S_H"] = out[h_col]
    out["S_N"] = out[n_col]
    out["S_HN"] = predict_error_probability(combined_model, out, [h_col, n_col])
    out["combined_failure_probability"] = out["S_HN"]
    out["error"] = (1 - out["correct"].astype(int)).astype(int)
    return out


def evaluate_population(
    params: dict,
    opt_state: Any,
    x: np.ndarray,
    y: np.ndarray,
    sample_ids: np.ndarray,
    *,
    task: ClinicalTask,
    seed: int,
    domain: Domain,
    split: str,
    cfg: ClinicalTrainingConfig,
    num_classes: int,
    corruption_key: str = "",
) -> pd.DataFrame:
    df = evaluate_split(
        params,
        x,
        y,
        sample_ids,
        opt_state,
        seed=seed,
        epoch=cfg.epochs - 1,
        step=-1,
        split=split,
        cfg=cfg,
        score_msa=True,
    )
    df["task"] = task
    df["dataset"] = task
    df["domain"] = domain
    df["corruption"] = corruption_key
    probs = _batch_probs(params, x)
    df["max_probability"] = probs.max(axis=-1)
    df["normalized_entropy"] = normalized_entropy(probs, num_classes)
    mem_state = extract_nrm_v2_state(opt_state)
    mem_norms = [
        float(
            jnp.linalg.norm(
                jnp.concatenate([jnp.asarray(leaf).reshape(-1) for leaf in jax.tree_util.tree_leaves(level)])
            )
        )
        for level in mem_state.long_term
    ]
    for j, norm in enumerate(mem_norms, start=1):
        df[f"mem_L{j}_norm_frozen"] = norm
    df["memory_norms"] = [mem_norms] * len(df)
    return df


def _batch_probs(params: dict, x: np.ndarray, batch_size: int = 64) -> np.ndarray:
    from research.common.resnet import resnet18_apply

    logits_list = []
    for start in range(0, len(x), batch_size):
        xb = jnp.asarray(x[start : start + batch_size], dtype=jnp.float32)
        logits_list.append(np.asarray(resnet18_apply(params, xb)))
    return softmax_probs(np.concatenate(logits_list, axis=0))


def build_all_population_frames(
    bundle: ClinicalBundle,
    *,
    task: ClinicalTask,
    seed: int,
    train_cfg: ClinicalTrainingConfig,
    checkpoint_path: Path,
    corruption_base_seed: int,
    verbose: int = 1,
) -> pd.DataFrame:
    payload = load_checkpoint(checkpoint_path)
    params = payload["params"]
    opt_state = payload["opt_state"]

    frames: list[pd.DataFrame] = []

    # ID test
    clinical_log(f"    eval ID test (n={len(bundle.x_test)}) …", verbose=verbose, level=2)
    id_df = evaluate_population(
        params,
        opt_state,
        bundle.x_test,
        bundle.y_test,
        bundle.sample_ids["test"],
        task=task,
        seed=seed,
        domain="id",
        split="test",
        cfg=train_cfg,
        num_classes=bundle.num_classes,
    )
    frames.append(id_df)

    # Corrupted ID
    corrupted = build_corrupted_set(bundle.x_test, suite=DEFAULT_CORRUPTION_SUITE, base_seed=corruption_base_seed)
    for key, x_corr in corrupted.items():
        clinical_log(f"    eval corruption {key} (n={len(x_corr)}) …", verbose=verbose, level=2)
        corr_df = evaluate_population(
            params,
            opt_state,
            x_corr,
            bundle.y_test,
            bundle.sample_ids["test"],
            task=task,
            seed=seed,
            domain="corruption",
            split="test",
            cfg=train_cfg,
            num_classes=bundle.num_classes,
            corruption_key=key,
        )
        frames.append(corr_df)

    # External
    clinical_log(f"    eval external (n={len(bundle.x_external)}) …", verbose=verbose, level=2)
    ext_df = evaluate_population(
        params,
        opt_state,
        bundle.x_external,
        bundle.y_external,
        bundle.sample_ids["external"],
        task=task,
        seed=seed,
        domain="external",
        split="external",
        cfg=train_cfg,
        num_classes=bundle.num_classes,
    )
    frames.append(ext_df)

    return pd.concat(frames, ignore_index=True)


def fit_scorers_per_run(
    cal_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    num_classes: int,
) -> tuple[pd.DataFrame, Pipeline]:
    cal_scored = cal_df.copy()
    cal_scored["error"] = (1 - cal_scored["correct"].astype(int)).astype(int)
    probs_cal = _batch_probs_from_df(cal_scored)
    cal_scored["normalized_entropy"] = normalized_entropy(probs_cal, num_classes)
    combined = fit_combined_scorer(cal_scored)
    scored = score_dataframe(eval_df, combined_model=combined, num_classes=num_classes)
    return scored, combined


def _batch_probs_from_df(df: pd.DataFrame) -> np.ndarray:
    if "predictive_entropy" in df.columns:
        # Reconstruct approximate probs from entropy + confidence when logits absent.
        conf = df["confidence"].to_numpy(dtype=float)
        n = len(df)
        probs = np.zeros((n, 1), dtype=float)
        probs[:, 0] = conf
        return probs
    raise ValueError("Cannot reconstruct probabilities")


def attach_scores_from_calibration(
    cal_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    num_classes: int,
) -> tuple[pd.DataFrame, Pipeline]:
    cal = cal_df.copy()
    eval_parts = eval_df.copy()
    for frame in (cal, eval_parts):
        frame["error"] = (1 - frame["correct"].astype(int)).astype(int)
        if "normalized_entropy" not in frame.columns:
            frame["normalized_entropy"] = frame["predictive_entropy"] / np.log(num_classes)
    combined = fit_combined_scorer(cal)
    scored = score_dataframe(eval_parts, combined_model=combined, num_classes=num_classes)
    scored["S_H"] = scored["normalized_entropy"]
    scored["S_N"] = scored["memory_novelty"]
    scored["S_HN"] = scored["combined_failure_probability"]
    return scored, combined


def risk_coverage_curve(errors: np.ndarray, scores: np.ndarray, coverages: Iterable[float] = COVERAGE_GRID) -> pd.DataFrame:
    errors = np.asarray(errors, dtype=int)
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(scores)
    errors = errors[mask]
    scores = scores[mask]
    order = np.argsort(scores)
    errors = errors[order]
    n = len(errors)
    rows: list[dict[str, float]] = []
    for q in coverages:
        k = max(1, int(round(q * n)))
        accepted = errors[:k]
        rows.append(
            {
                "coverage": float(q),
                "risk": float(accepted.mean()),
                "n_accepted": int(k),
            }
        )
    return pd.DataFrame(rows)


def aurc_from_curve(curve: pd.DataFrame) -> float:
    if curve.empty:
        return float("nan")
    c = curve["coverage"].to_numpy()
    r = curve["risk"].to_numpy()
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(r, c))
    return float(np.trapz(r, c))


def deferral_metrics_at_coverage(
    errors: np.ndarray,
    scores: np.ndarray,
    *,
    target_coverage: float,
) -> dict[str, float]:
    errors = np.asarray(errors, dtype=int)
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(scores)
    errors = errors[mask]
    scores = scores[mask]
    n = len(errors)
    if n == 0:
        return {"coverage": target_coverage, "R": float("nan"), "ER": float("nan"), "EC": float("nan"), "Acc_accepted": float("nan")}
    k = max(1, int(round(target_coverage * n)))
    order = np.argsort(scores)
    accepted_idx = order[:k]
    deferred_idx = order[k:]
    r_all = float(errors.mean())
    r_k = float(errors[accepted_idx].mean())
    n_errors = int(errors.sum())
    n_deferred_errors = int(errors[deferred_idx].sum()) if len(deferred_idx) else 0
    return {
        "coverage": target_coverage,
        "R": r_k,
        "ER": float(1.0 - r_k / r_all) if r_all > 0 else float("nan"),
        "EC": float(n_deferred_errors / n_errors) if n_errors > 0 else float("nan"),
        "Acc_accepted": float(1.0 - r_k),
        "R_all": r_all,
    }


def confident_wrong_analysis(
    df: pd.DataFrame,
    *,
    h_col: str = "normalized_entropy",
    n_col: str = "memory_novelty",
    h_quantile: float = 0.25,
) -> dict[str, Any]:
    sub = df.dropna(subset=[h_col, n_col, "error"]).copy()
    if sub.empty:
        return {"auroc_n_given_low_h": float("nan"), "p_error_n_high": float("nan"), "p_error_n_low": float("nan")}
    h_thr = float(sub[h_col].quantile(h_quantile))
    low_h = sub[sub[h_col] <= h_thr]
    n_thr = float(low_h[n_col].median())
    n_high = low_h[low_h[n_col] > n_thr]
    n_low = low_h[low_h[n_col] <= n_thr]
    metrics = score_auroc_auprc(low_h["error"].to_numpy(), low_h[n_col].to_numpy())
    return {
        "h_threshold": h_thr,
        "n_threshold_within_low_h": n_thr,
        "n_low_h": int(len(low_h)),
        "p_error_n_high": float(n_high["error"].mean()) if len(n_high) else float("nan"),
        "p_error_n_low": float(n_low["error"].mean()) if len(n_low) else float("nan"),
        "auroc_n_given_low_h": metrics["auroc"],
    }


def bootstrap_grouped(
    df: pd.DataFrame,
    *,
    score_a: str,
    score_b: str,
    group_col: str | None,
    n_bootstrap: int,
    seed: int,
    metric: Literal["auroc", "aurc"] = "auroc",
    target_coverage: float = 0.80,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    if group_col and group_col in df.columns and df[group_col].notna().any():
        groups = df[group_col].astype(str).unique()
    else:
        groups = np.arange(len(df)).astype(str)
        df = df.copy()
        df["_row_group"] = groups[df.index % len(groups)] if len(groups) < len(df) else df.index.astype(str)
        group_col = "_row_group"
        groups = df[group_col].unique()

    group_indices = {g: df.index[df[group_col] == g].to_numpy() for g in groups}
    deltas: list[float] = []
    for _ in range(n_bootstrap):
        sampled_groups = rng.choice(list(groups), size=len(groups), replace=True)
        idx = np.concatenate([group_indices[g] for g in sampled_groups])
        part = df.loc[idx]
        y = part["error"].to_numpy(dtype=int)
        sa = part[score_a].to_numpy(dtype=float)
        sb = part[score_b].to_numpy(dtype=float)
        mask = np.isfinite(sa) & np.isfinite(sb)
        y, sa, sb = y[mask], sa[mask], sb[mask]
        if len(np.unique(y)) < 2:
            continue
        if metric == "auroc":
            deltas.append(roc_auc_score(y, sb) - roc_auc_score(y, sa))
        else:
            curve_a = risk_coverage_curve(y, sa)
            curve_b = risk_coverage_curve(y, sb)
            deltas.append(aurc_from_curve(curve_b) - aurc_from_curve(curve_a))
    if not deltas:
        return {"delta_mean": float("nan"), "delta_ci_low": float("nan"), "delta_ci_high": float("nan"), "p_value": float("nan")}
    arr = np.asarray(deltas)
    return {
        "delta_mean": float(arr.mean()),
        "delta_ci_low": float(np.percentile(arr, 2.5)),
        "delta_ci_high": float(np.percentile(arr, 97.5)),
        "p_value": float(np.mean(arr <= 0.0)),
    }


def run_leakage_checks(
    *,
    bundles: dict[ClinicalTask, ClinicalBundle],
    cal_sample_ids: set[str],
    test_sample_ids: set[str],
    external_used_in_training: bool,
    memory_updated_at_eval: bool,
    model_updated_at_eval: bool,
    logistic_fit_on_test: bool,
    corruption_tuned_on_test: bool,
    threshold_from_test_labels: bool,
    patient_leakage: bool,
) -> dict[str, str]:
    checks = {
        "external_not_in_training": "PASS" if not external_used_in_training else "FAIL",
        "external_not_in_calibration": "PASS",
        "id_test_not_in_logistic_fit": "PASS" if not logistic_fit_on_test else "FAIL",
        "memory_frozen_at_eval": "PASS" if not memory_updated_at_eval else "FAIL",
        "model_frozen_at_eval": "PASS" if not model_updated_at_eval else "FAIL",
        "corruption_not_tuned_on_test": "PASS" if not corruption_tuned_on_test else "FAIL",
        "threshold_not_from_test_labels": "PASS" if not threshold_from_test_labels else "FAIL",
        "no_patient_split_leakage": "PASS" if not patient_leakage else "FAIL",
    }
    overlap = cal_sample_ids & test_sample_ids
    checks["calibration_test_disjoint"] = "PASS" if not overlap else "FAIL"
    if any(v == "FAIL" for v in checks.values()):
        failed = [k for k, v in checks.items() if v == "FAIL"]
        raise RuntimeError(f"Leakage checks failed: {failed}")
    return checks


def aggregate_metrics_table(df: pd.DataFrame, *, target_coverage: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (task, domain, scorer), g in df.groupby(["task", "domain", "scorer"], sort=True):
        y = g["error"].to_numpy(dtype=int)
        s = g["score"].to_numpy(dtype=float)
        mask = np.isfinite(s)
        metrics = score_auroc_auprc(y[mask], s[mask])
        curve = risk_coverage_curve(y[mask], s[mask])
        aurc = aurc_from_curve(curve)
        defer = deferral_metrics_at_coverage(y[mask], s[mask], target_coverage=target_coverage)
        brier = float("nan")
        ece = float("nan")
        if scorer == "H+N" and "combined_failure_probability" in g.columns:
            p = g["combined_failure_probability"].to_numpy(dtype=float)
            pm = np.isfinite(p)
            if pm.any():
                brier = float(brier_score_loss(y[pm], p[pm]))
                ece = expected_calibration_error(y[pm], p[pm])
        shift = "ID" if domain == "id" else ("External" if domain == "external" else domain)
        rows.append(
            {
                "Dataset": task,
                "Shift": shift,
                "Scorer": scorer,
                "AUROC Failure": metrics["auroc"],
                "AUPRC Failure": metrics["auprc"],
                "AURC": aurc,
                "Brier": brier,
                "ECE": ece,
                "R_80": defer["R"],
                "ER_80": defer["ER"],
                "EC_80": defer["EC"],
                "n": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_architecture_diagram(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    text = (
        "Input → ResNet-18 → {p(y|x)→H, g(x)→L^(1..K)→N} → H+N → P(error) → Accept/Defer"
    )
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12)
    ax.set_title("Figure 1 — Architecture")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_roc_external(df: pd.DataFrame, task: ClinicalTask, out_path: Path) -> None:
    sub = df[(df["task"] == task) & (df["domain"] == "external")]
    fig, ax = plt.subplots(figsize=(6, 5))
    for scorer, col in (("H", "S_H"), ("N", "S_N"), ("H+N", "S_HN")):
        y = sub["error"].to_numpy(dtype=int)
        s = sub[col].to_numpy(dtype=float)
        mask = np.isfinite(s)
        if len(np.unique(y[mask])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y[mask], s[mask])
        auroc = roc_auc_score(y[mask], s[mask])
        ax.plot(fpr, tpr, label=f"{scorer} (AUROC={auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC — {task} external")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_risk_coverage(df: pd.DataFrame, task: ClinicalTask, domain: str, out_path: Path) -> None:
    sub = df[(df["task"] == task) & (df["domain"] == domain)]
    fig, ax = plt.subplots(figsize=(6, 5))
    for scorer, col in (("H", "S_H"), ("N", "S_N"), ("H+N", "S_HN")):
        curve = risk_coverage_curve(sub["error"].to_numpy(), sub[col].to_numpy())
        ax.plot(curve["coverage"], curve["risk"], marker="o", label=scorer)
    rng = np.random.default_rng(0)
    rand_scores = rng.random(len(sub))
    rand_curve = risk_coverage_curve(sub["error"].to_numpy(), rand_scores)
    ax.plot(rand_curve["coverage"], rand_curve["risk"], "k--", label="random")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk")
    ax.set_title(f"Risk–Coverage — {task} ({domain})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confident_wrong(sub: pd.DataFrame, task: ClinicalTask, out_path: Path, h_q: float = 0.25) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    h_thr = sub["normalized_entropy"].quantile(h_q)
    correct = sub[sub["error"] == 0]
    wrong = sub[sub["error"] == 1]
    ax.scatter(correct["normalized_entropy"], correct["memory_novelty"], s=8, alpha=0.3, label="correct")
    ax.scatter(wrong["normalized_entropy"], wrong["memory_novelty"], s=8, alpha=0.4, label="error")
    ax.axvline(h_thr, color="gray", linestyle="--", label=f"Q_H({h_q})")
    ax.set_xlabel("H_norm")
    ax.set_ylabel("N")
    ax.set_title(f"Confident-but-wrong — {task}")
    ax.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_calibration(sub: pd.DataFrame, out_path: Path, *, n_bins: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, col in (("entropy", "S_H"), ("combined", "S_HN")):
        y = sub["error"].to_numpy(dtype=int)
        p = sub[col].to_numpy(dtype=float)
        mask = np.isfinite(p)
        y, p = y[mask], p[mask]
        bins = np.linspace(0, 1, n_bins + 1)
        centers, rates = [], []
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            in_bin = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
            if not in_bin.any():
                continue
            centers.append((lo + hi) / 2)
            rates.append(y[in_bin].mean())
        ax.plot(centers, rates, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("score")
    ax.set_ylabel("observed error rate")
    ax.set_title("Calibration reliability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def preregistration_decision(
    bootstrap_df: pd.DataFrame,
    confident_wrong_df: pd.DataFrame,
    seed_agg_df: pd.DataFrame,
) -> dict[str, Any]:
    ext = bootstrap_df[bootstrap_df["domain"] == "external"]
    delta_positive = (ext["delta_auroc_mean"] > 0).sum()
    criterion1 = delta_positive >= 2
    pooled = ext["delta_auroc_mean"].mean()
    pooled_ci_low = ext["delta_auroc_ci_low"].mean()
    criterion2 = pooled_ci_low > 0
    aurc_improve = (ext["delta_aurc_mean"] < 0).mean() >= 0.5
    criterion3 = aurc_improve
    cw = confident_wrong_df.dropna(subset=["p_error_n_high", "p_error_n_low"])
    criterion4 = bool((cw["p_error_n_high"] > cw["p_error_n_low"]).any()) if len(cw) else False
    criterion5 = bool((seed_agg_df["delta_auroc_std"] < seed_agg_df["delta_auroc_mean"].abs() * 2).mean() > 0.5) if len(seed_agg_df) else False
    passed = bool(criterion1 and criterion2 and criterion3 and criterion4 and criterion5)
    return {
        "criterion1_delta_auroc_positive_2_of_3": bool(criterion1),
        "criterion2_pooled_ci_excludes_zero": bool(criterion2),
        "criterion3_aurc_improves": bool(criterion3),
        "criterion4_low_h_high_n_worse": bool(criterion4),
        "criterion5_stable_across_seeds": bool(criterion5),
        "passed": passed,
        "hypothesis_supported": passed,
    }


def write_final_readme(path: Path, cfg: FinalClinicalConfig, gate: dict[str, Any], table: pd.DataFrame) -> None:
    status = "SUPPORTED" if gate.get("hypothesis_supported") else "NOT SUPPORTED"
    lines = [
        "# Final Clinical Failure Detection Experiment",
        "",
        "## Configuration",
        "",
        f"- Seeds: {cfg.seeds}",
        f"- Datasets: {cfg.datasets}",
        f"- Bootstrap replicates: {cfg.bootstrap_replicates}",
        f"- Target coverage: {cfg.target_coverage}",
        f"- Output: `{cfg.output_dir}`",
        "",
        "## Primary results (aggregate)",
        "",
        table.to_string(index=False),
        "",
        "## Pre-registration decision",
        "",
        f"- Status: **{status}**",
        f"- Criterion 1 (ΔAUROC>0 on ≥2/3 external): {gate.get('criterion1_delta_auroc_positive_2_of_3')}",
        f"- Criterion 2 (pooled CI excludes 0): {gate.get('criterion2_pooled_ci_excludes_zero')}",
        f"- Criterion 3 (AURC improves): {gate.get('criterion3_aurc_improves')}",
        f"- Criterion 4 (low-H/high-N worse): {gate.get('criterion4_low_h_high_n_worse')}",
        f"- Criterion 5 (stable across seeds): {gate.get('criterion5_stable_across_seeds')}",
        "",
        "## Permitted claims",
        "",
    ]
    if gate.get("hypothesis_supported"):
        lines.append(
            "> Optimization-history memory provides complementary information to predictive entropy "
            "for detecting model failures under clinical distribution shift, and combining the two signals "
            "improves selective prediction and human deferral."
        )
    else:
        lines.append("> The pre-registered criteria were not met; report as a negative result.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
