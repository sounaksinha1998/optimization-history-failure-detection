"""Final NRO experiment: test I(E; N | H) > 0 on DermaMNIST with frozen checkpoints."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline

from research.common.error_prediction import (
    bootstrap_auroc_delta,
    fit_failure_scorer,
    pipeline_logistic_coefficients,
    predict_error_probability,
    score_auroc_auprc,
)

EvaluationDomain = Literal["external", "id", "corruption"]


@dataclass
class NROFinalConfig:
    """Configuration for the final NRO conditional-information experiment."""

    repo_root: Path | None = None
    source_experiment_dir: Path | None = None
    output_dir: Path | None = None
    seeds: tuple[int, ...] = (42, 123, 456)
    task: str = "dermamnist"
    evaluation_domain: EvaluationDomain = "external"
    h_col: str = "normalized_entropy"
    n_col: str = "memory_novelty"
    num_classes: int = 7
    n_bootstrap: int = 2000
    bootstrap_seed: int = 20260829
    n_h_bins: int = 20
    n_entropy_bins: int = 5
    permutation_seed: int = 42
    ci_level: float = 0.95
    prob_epsilon: float = 1e-6
    logistic_max_iter: int = 2000
    show_plots: bool = True
    save_artifacts: bool = True
    aggregate_seeds: bool = True

    def resolve_paths(self, cwd: Path | None = None) -> NROFinalConfig:
        root = self.repo_root
        if root is None:
            root = Path(cwd or Path.cwd()).resolve()
            if not (root / "research").exists() and (root.parent / "research").exists():
                root = root.parent
        self.repo_root = root
        if self.source_experiment_dir is None:
            self.source_experiment_dir = root / "research" / "final_experiment"
        if self.output_dir is None:
            self.output_dir = root / "research" / "dermamnist_nro_final"
        return self


def _clip_probs(p: np.ndarray, eps: float) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)


def _binary_log_loss(y: np.ndarray, p: np.ndarray, *, eps: float) -> float:
    y = np.asarray(y, dtype=int)
    p = _clip_probs(p, eps)
    return float(log_loss(y, p, labels=[0, 1]))


def load_calibration_frames(cfg: NROFinalConfig) -> pd.DataFrame:
    source = cfg.source_experiment_dir
    if source is None:
        raise ValueError("source_experiment_dir is required")
    parts: list[pd.DataFrame] = []
    for seed in cfg.seeds:
        path = source / "calibration" / f"{cfg.task}_seed{seed}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing calibration CSV: {path}")
        parts.append(pd.read_csv(path))
    return pd.concat(parts, ignore_index=True)


def load_test_frames(cfg: NROFinalConfig) -> pd.DataFrame:
    source = cfg.source_experiment_dir
    if source is None:
        raise ValueError("source_experiment_dir is required")
    parts: list[pd.DataFrame] = []
    for seed in cfg.seeds:
        path = source / "scored" / f"{cfg.task}_seed{seed}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing scored CSV: {path}")
        df = pd.read_csv(path)
        parts.append(df[df["domain"] == cfg.evaluation_domain].copy())
    out = pd.concat(parts, ignore_index=True)
    if out.empty:
        raise ValueError(f"No rows for evaluation_domain={cfg.evaluation_domain!r}")
    return out


def build_per_sample_table(df: pd.DataFrame, cfg: NROFinalConfig) -> pd.DataFrame:
    required = {"correct", cfg.n_col, "sample_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    out = df.copy()
    if cfg.h_col not in out.columns:
        if "predictive_entropy" not in out.columns:
            raise ValueError(f"Missing H column {cfg.h_col!r} and cannot derive from predictive_entropy")
        out[cfg.h_col] = out["predictive_entropy"].astype(float) / np.log(cfg.num_classes)
    out["H"] = out[cfg.h_col].astype(float)
    out["N"] = out[cfg.n_col].astype(float)
    out["E"] = (1 - pd.to_numeric(out["correct"], errors="coerce").fillna(0)).astype(int)
    out["error"] = out["E"]
    if "predictive_entropy" in out.columns:
        out["H_raw"] = out["predictive_entropy"].astype(float)
    return out


def assert_no_leakage(cal_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    cal_ids = set(cal_df["sample_id"].astype(str))
    test_ids = set(test_df["sample_id"].astype(str))
    overlap = cal_ids & test_ids
    assertions = {
        "calibration_n": len(cal_df),
        "test_n": len(test_df),
        "overlap_n": len(overlap),
        "disjoint": len(overlap) == 0,
        "cal_split_all_calibration": bool((cal_df.get("split", pd.Series(dtype=str)) == "calibration").all())
        if "split" in cal_df.columns
        else None,
    }
    if overlap:
        raise AssertionError(f"Calibration/test sample_id leakage: {len(overlap)} overlapping ids")
    return assertions


def assign_h_bins(
    df: pd.DataFrame,
    *,
    bin_edges: np.ndarray,
) -> pd.Series:
    # Right-inclusive last bin via labels=False
    return pd.cut(df["H"], bins=bin_edges, include_lowest=True, labels=False)


def compute_h_bin_edges(
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    n_bins: int,
) -> np.ndarray:
    pooled_h = np.concatenate([cal_df["H"].to_numpy(), test_df["H"].to_numpy()])
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(pooled_h, quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        edges = np.linspace(float(pooled_h.min()), float(pooled_h.max()), n_bins + 1)
    return edges


def permute_n_within_h_bins(
    df: pd.DataFrame,
    *,
    bin_edges: np.ndarray,
    seed: int,
) -> np.ndarray:
    out = df["N"].to_numpy(dtype=float).copy()
    h_bin = assign_h_bins(df, bin_edges=bin_edges)
    rng = np.random.default_rng(seed)
    for bin_id in sorted(h_bin.dropna().unique()):
        mask = (h_bin == bin_id).to_numpy()
        idx = np.where(mask)[0]
        if len(idx) <= 1:
            continue
        permuted = out[idx].copy()
        rng.shuffle(permuted)
        out[idx] = permuted
    return out


def fit_failure_models(
    cal_df: pd.DataFrame,
    *,
    n_perm: np.ndarray | None = None,
    max_iter: int = 2000,
) -> tuple[Pipeline, Pipeline, dict[str, Any]]:
    fit_df = cal_df.copy()
    if n_perm is not None:
        fit_df["N"] = n_perm
    y = fit_df["E"].to_numpy(dtype=int)

    model_h = fit_failure_scorer(fit_df, ["H"], error_col="E", max_iter=max_iter)
    model_hn = fit_failure_scorer(fit_df, ["H", "N"], error_col="E", max_iter=max_iter)

    meta = {
        "h_coefficients": pipeline_logistic_coefficients(model_h, ["H"]),
        "hn_coefficients": pipeline_logistic_coefficients(model_hn, ["H", "N"]),
        "cal_n": int(len(cal_df)),
        "cal_error_rate": float(y.mean()),
        "permuted_n": n_perm is not None,
    }
    return model_h, model_hn, meta


def _failure_probs(
    test_df: pd.DataFrame,
    model_h: Pipeline,
    model_hn: Pipeline,
    *,
    n_perm: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    h_df = test_df[["H"]].copy()
    hn_df = test_df[["H", "N"]].copy()
    if n_perm is not None:
        hn_df["N"] = n_perm
    q_h = predict_error_probability(model_h, h_df, ["H"])
    q_hn = predict_error_probability(model_hn, hn_df, ["H", "N"])
    return q_h, q_hn


def evaluate_models_on_test(
    test_df: pd.DataFrame,
    model_h: Pipeline,
    model_hn: Pipeline,
    *,
    n_perm: np.ndarray | None = None,
    eps: float,
) -> dict[str, float]:
    y = test_df["E"].to_numpy(dtype=int)
    q_h, q_hn = _failure_probs(test_df, model_h, model_hn, n_perm=n_perm)

    l_h = _binary_log_loss(y, q_h, eps=eps)
    l_hn = _binary_log_loss(y, q_hn, eps=eps)
    auroc_h = score_auroc_auprc(y, test_df["H"].to_numpy(dtype=float))["auroc"]
    auroc_hn = score_auroc_auprc(y, q_hn)["auroc"]

    return {
        "L_H": l_h,
        "L_HN": l_hn,
        "delta_L": l_h - l_hn,
        "auroc_H": auroc_h,
        "auroc_HN": auroc_hn,
        "delta_auroc": auroc_hn - auroc_h,
        "test_n": int(len(test_df)),
        "test_error_rate": float(y.mean()),
    }


def bootstrap_test_metrics(
    test_df: pd.DataFrame,
    model_h: Pipeline,
    model_hn: Pipeline,
    model_hn_perm: Pipeline,
    *,
    n_perm_test: np.ndarray,
    cfg: NROFinalConfig,
) -> pd.DataFrame:
    y = test_df["E"].to_numpy(dtype=int)
    q_h_all, q_hn_all = _failure_probs(test_df, model_h, model_hn)
    _, q_hn_perm_all = _failure_probs(test_df, model_h, model_hn_perm, n_perm=n_perm_test)
    h_all = test_df["H"].to_numpy(dtype=float)
    n_rows = len(test_df)

    rng = np.random.default_rng(cfg.bootstrap_seed)
    rows: list[dict[str, float]] = []
    for b in range(cfg.n_bootstrap):
        idx = rng.integers(0, n_rows, size=n_rows)
        y_b = y[idx]
        q_h = q_h_all[idx]
        q_hn = q_hn_all[idx]
        q_hn_perm = q_hn_perm_all[idx]

        l_h = _binary_log_loss(y_b, q_h, eps=cfg.prob_epsilon)
        l_hn = _binary_log_loss(y_b, q_hn, eps=cfg.prob_epsilon)
        l_hn_perm = _binary_log_loss(y_b, q_hn_perm, eps=cfg.prob_epsilon)

        auroc_h = score_auroc_auprc(y_b, h_all[idx])["auroc"]
        auroc_hn = score_auroc_auprc(y_b, q_hn_all[idx])["auroc"]

        rows.append(
            {
                "bootstrap_id": b,
                "delta_L": l_h - l_hn,
                "delta_L_perm": l_hn_perm - l_hn,
                "delta_auroc": auroc_hn - auroc_h if np.isfinite(auroc_h) and np.isfinite(auroc_hn) else float("nan"),
                "L_H": l_h,
                "L_HN": l_hn,
                "L_HN_perm": l_hn_perm,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, *, ci_level: float) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    alpha = (1.0 - ci_level) / 2.0
    return (
        float(np.mean(values)),
        float(np.percentile(values, 100 * alpha)),
        float(np.percentile(values, 100 * (1 - alpha))),
    )


def stratified_n_auroc(test_df: pd.DataFrame, *, n_bins: int) -> pd.DataFrame:
    df = test_df.dropna(subset=["H", "N", "E"]).copy()
    if df.empty:
        return pd.DataFrame()
    try:
        df["entropy_bin"] = pd.qcut(df["H"], q=n_bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for bin_label, group in df.groupby("entropy_bin", observed=True):
        y = group["E"].to_numpy(dtype=int)
        scores = group["N"].to_numpy(dtype=float)
        metrics = score_auroc_auprc(y, scores)
        rows.append(
            {
                "entropy_bin": str(bin_label),
                "h_min": float(group["H"].min()),
                "h_max": float(group["H"].max()),
                "n": len(group),
                "auroc_n": metrics["auroc"],
                "auprc_n": metrics["auprc"],
                "error_rate": float(y.mean()),
                "auroc_above_half": bool(metrics["auroc"] > 0.5) if np.isfinite(metrics["auroc"]) else False,
            }
        )
    return pd.DataFrame(rows)


def matched_high_low_n_analysis(
    test_df: pd.DataFrame,
    *,
    n_h_bins: int = 10,
    n_quantile: float = 0.25,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 0,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Match high-N vs low-N samples with approximately equal H within entropy bins."""
    df = test_df.dropna(subset=["H", "N", "E"]).copy()
    rows: list[dict[str, Any]] = []
    try:
        df["h_bin"] = pd.qcut(df["H"], q=n_h_bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame()

    for bin_label, group in df.groupby("h_bin", observed=True):
        if len(group) < 8:
            continue
        n_low_thr = group["N"].quantile(n_quantile)
        n_high_thr = group["N"].quantile(1.0 - n_quantile)
        low = group[group["N"] <= n_low_thr].copy()
        high = group[group["N"] >= n_high_thr].copy()
        if low.empty or high.empty:
            continue

        # Greedy nearest-H matching without replacement
        low = low.sort_values("H").reset_index(drop=True)
        high = high.sort_values("H").reset_index(drop=True)
        n_pairs = min(len(low), len(high))
        low = low.iloc[:n_pairs]
        high = high.iloc[:n_pairs]
        high_errors = high["E"].to_numpy(dtype=int)
        low_errors = low["E"].to_numpy(dtype=int)
        delta = float(high_errors.mean() - low_errors.mean())

        rng = np.random.default_rng(bootstrap_seed + hash(str(bin_label)) % 10_000)
        boot_deltas: list[float] = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n_pairs, size=n_pairs)
            boot_deltas.append(float(high_errors[idx].mean() - low_errors[idx].mean()))
        alpha = (1.0 - ci_level) / 2.0
        rows.append(
            {
                "h_bin": str(bin_label),
                "h_min": float(group["H"].min()),
                "h_max": float(group["H"].max()),
                "n_pairs": n_pairs,
                "failure_rate_high_n": float(high_errors.mean()),
                "failure_rate_low_n": float(low_errors.mean()),
                "delta_failure_rate": delta,
                "delta_ci_low": float(np.percentile(boot_deltas, 100 * alpha)),
                "delta_ci_high": float(np.percentile(boot_deltas, 100 * (1 - alpha))),
            }
        )
    return pd.DataFrame(rows)


def preregistered_decision(
    results: dict[str, Any],
    stratified_df: pd.DataFrame,
    matched_df: pd.DataFrame,
) -> dict[str, Any]:
    primary_dl = results["delta_L"] > 0 and results["delta_L_ci_excludes_zero"]
    perm_dl = results["delta_L_perm"] > 0 and results["delta_L_perm_ci_excludes_zero"]
    secondary_auroc = results["delta_auroc"] > 0
    strata_majority = False
    if len(stratified_df) > 0 and "auroc_above_half" in stratified_df.columns:
        strata_majority = bool(stratified_df["auroc_above_half"].mean() > 0.5)
    matched_positive = False
    if len(matched_df) > 0:
        matched_positive = bool((matched_df["delta_failure_rate"] > 0).mean() > 0.5)

    passed = bool(primary_dl and perm_dl)
    return {
        "primary_delta_L_positive": bool(results["delta_L"] > 0),
        "primary_delta_L_ci_excludes_zero": bool(results["delta_L_ci_excludes_zero"]),
        "permutation_delta_L_perm_positive": bool(results["delta_L_perm"] > 0),
        "permutation_delta_L_perm_ci_excludes_zero": bool(results["delta_L_perm_ci_excludes_zero"]),
        "secondary_auroc_improves": bool(secondary_auroc),
        "secondary_strata_majority_auroc_n_above_half": bool(strata_majority),
        "secondary_matched_high_n_higher_failure_rate": bool(matched_positive),
        "passed": passed,
        "hypothesis_supported": passed,
        "answers_I_E_N_given_H_positive": passed,
    }


def _checkpoint_paths(cfg: NROFinalConfig) -> dict[str, str]:
    source = cfg.source_experiment_dir
    if source is None:
        return {}
    out: dict[str, str] = {}
    for seed in cfg.seeds:
        p = source / "checkpoints" / f"{cfg.task}_seed{seed}.pkl"
        out[f"seed_{seed}"] = str(p.resolve())
    return out


def _ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "audit": output_dir / "audit",
        "figures": output_dir / "figures",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def plot_failure_probability_curves(
    test_df: pd.DataFrame,
    model_h: Pipeline,
    model_hn: Pipeline,
    out_path: Path,
) -> None:
    h_grid = np.linspace(test_df["H"].min(), test_df["H"].max(), 100)
    n_median = float(test_df["N"].median())
    h_df = pd.DataFrame({"H": h_grid})
    hn_df = pd.DataFrame({"H": h_grid, "N": np.full_like(h_grid, n_median)})
    q_h = predict_error_probability(model_h, h_df, ["H"])
    q_hn = predict_error_probability(model_hn, hn_df, ["H", "N"])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(h_grid, q_h, label="P(E|H)", linewidth=2)
    ax.plot(h_grid, q_hn, label=f"P(E|H,N) at N=median ({n_median:.3f})", linewidth=2)
    ax.set_xlabel("H (normalized entropy)")
    ax.set_ylabel("Predicted failure probability")
    ax.set_title("Calibrated failure probability vs H")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_delta_logloss_bootstrap(boot_df: pd.DataFrame, out_path: Path, *, ci_level: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in zip(
        axes,
        ["delta_L", "delta_L_perm"],
        ["ΔL = L_H − L_HN", "ΔL_perm = L_HN,perm − L_HN"],
    ):
        vals = boot_df[col].dropna().to_numpy()
        ax.hist(vals, bins=40, color="steelblue", alpha=0.85, edgecolor="white")
        mean, lo, hi = bootstrap_ci(vals, ci_level=ci_level)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.axvline(mean, color="crimson", linewidth=2, label=f"mean={mean:.4f}")
        ax.axvline(lo, color="orange", linestyle=":", linewidth=1.5)
        ax.axvline(hi, color="orange", linestyle=":", linewidth=1.5, label=f"95% CI [{lo:.4f}, {hi:.4f}]")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_h_n_scatter(test_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    correct = test_df[test_df["E"] == 0]
    wrong = test_df[test_df["E"] == 1]
    ax.scatter(correct["H"], correct["N"], alpha=0.25, s=10, c="steelblue", label="correct")
    ax.scatter(wrong["H"], wrong["N"], alpha=0.5, s=12, c="crimson", label="error")
    ax.set_xlabel("H (normalized entropy)")
    ax.set_ylabel("N (memory novelty)")
    ax.set_title("H vs N colored by failure")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_stratified_auroc(stratified_df: pd.DataFrame, out_path: Path) -> None:
    if stratified_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(stratified_df))
    ax.bar(x, stratified_df["auroc_n"], color="teal", alpha=0.85)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels([f"bin {i+1}" for i in x], rotation=0)
    ax.set_ylabel("AUROC(E, N)")
    ax.set_title("Novelty AUROC within entropy strata")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_matched_failure_rates(matched_df: pd.DataFrame, out_path: Path) -> None:
    if matched_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(matched_df))
    width = 0.35
    ax.bar(x - width / 2, matched_df["failure_rate_low_n"], width, label="low N", color="steelblue")
    ax.bar(x + width / 2, matched_df["failure_rate_high_n"], width, label="high N", color="crimson")
    ax.set_xticks(x)
    ax.set_xticklabels([f"bin {i+1}" for i in x])
    ax.set_ylabel("Failure rate")
    ax.set_title("Matched high-N vs low-N failure rates (equal H bins)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_single_seed_analysis(
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: NROFinalConfig,
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    cal = build_per_sample_table(cal_df, cfg)
    test = build_per_sample_table(test_df, cfg)
    if seed is not None:
        cal = cal[cal["seed"] == seed].copy()
        test = test[test["seed"] == seed].copy()

    leakage = assert_no_leakage(cal, test)
    bin_edges = compute_h_bin_edges(cal, test, n_bins=cfg.n_h_bins)

    n_perm_cal = permute_n_within_h_bins(cal, bin_edges=bin_edges, seed=cfg.permutation_seed)
    n_perm_test = permute_n_within_h_bins(test, bin_edges=bin_edges, seed=cfg.permutation_seed + 1)

    model_h, model_hn, fit_meta = fit_failure_models(cal, max_iter=cfg.logistic_max_iter)
    _, model_hn_perm, _ = fit_failure_models(cal, n_perm=n_perm_cal, max_iter=cfg.logistic_max_iter)

    metrics = evaluate_models_on_test(test, model_h, model_hn, eps=cfg.prob_epsilon)
    metrics_perm = evaluate_models_on_test(test, model_h, model_hn_perm, n_perm=n_perm_test, eps=cfg.prob_epsilon)
    metrics["delta_L_perm"] = metrics_perm["L_HN"] - metrics["L_HN"]

    boot_df = bootstrap_test_metrics(
        test, model_h, model_hn, model_hn_perm, n_perm_test=n_perm_test, cfg=cfg
    )
    dl_mean, dl_lo, dl_hi = bootstrap_ci(boot_df["delta_L"].to_numpy(), ci_level=cfg.ci_level)
    dlp_mean, dlp_lo, dlp_hi = bootstrap_ci(boot_df["delta_L_perm"].to_numpy(), ci_level=cfg.ci_level)
    dau_mean, dau_lo, dau_hi = bootstrap_ci(boot_df["delta_auroc"].to_numpy(), ci_level=cfg.ci_level)

    stratified_df = stratified_n_auroc(test, n_bins=cfg.n_entropy_bins)
    matched_df = matched_high_low_n_analysis(
        test,
        n_h_bins=cfg.n_entropy_bins,
        n_bootstrap=cfg.n_bootstrap,
        bootstrap_seed=cfg.bootstrap_seed,
        ci_level=cfg.ci_level,
    )

    results = {
        **metrics,
        "delta_L_bootstrap_mean": dl_mean,
        "delta_L_ci_low": dl_lo,
        "delta_L_ci_high": dl_hi,
        "delta_L_ci_excludes_zero": bool(dl_lo > 0),
        "delta_L_perm_bootstrap_mean": dlp_mean,
        "delta_L_perm_ci_low": dlp_lo,
        "delta_L_perm_ci_high": dlp_hi,
        "delta_L_perm_ci_excludes_zero": bool(dlp_lo > 0),
        "delta_auroc_bootstrap_mean": dau_mean,
        "delta_auroc_ci_low": dau_lo,
        "delta_auroc_ci_high": dau_hi,
        "fit_meta": fit_meta,
        "leakage": leakage,
        "seed": seed,
    }
    gate = preregistered_decision(results, stratified_df, matched_df)
    results["decision_gate"] = gate

    per_sample = test.copy()
    q_h, q_hn = _failure_probs(per_sample, model_h, model_hn)
    per_sample["q_H"] = q_h
    per_sample["q_HN"] = q_hn
    per_sample["N_perm"] = n_perm_test
    _, q_hn_perm = _failure_probs(per_sample, model_h, model_hn_perm, n_perm=n_perm_test)
    per_sample["q_HN_perm"] = q_hn_perm

    return {
        "results": results,
        "bootstrap_df": boot_df,
        "stratified_df": stratified_df,
        "matched_df": matched_df,
        "per_sample_df": per_sample,
        "models": {"H": model_h, "HN": model_hn, "HN_perm": model_hn_perm},
        "bin_edges": bin_edges,
    }


def aggregate_pooled_from_seed_runs(
    per_seed_runs: dict[str, Any],
    cfg: NROFinalConfig,
) -> dict[str, Any]:
    """Pooled metrics using per-seed calibration fits (matches failure-detection pipeline)."""
    per_sample = pd.concat(
        [run["per_sample_df"] for run in per_seed_runs.values()],
        ignore_index=True,
    )
    y = per_sample["E"].to_numpy(dtype=int)
    q_h = per_sample["q_H"].to_numpy(dtype=float)
    q_hn = per_sample["q_HN"].to_numpy(dtype=float)
    q_hn_perm = per_sample["q_HN_perm"].to_numpy(dtype=float)
    h_raw = per_sample["H"].to_numpy(dtype=float)

    l_h = _binary_log_loss(y, q_h, eps=cfg.prob_epsilon)
    l_hn = _binary_log_loss(y, q_hn, eps=cfg.prob_epsilon)
    l_hn_perm = _binary_log_loss(y, q_hn_perm, eps=cfg.prob_epsilon)
    auroc_h = score_auroc_auprc(y, h_raw)["auroc"]
    auroc_hn = score_auroc_auprc(y, q_hn)["auroc"]

    rng = np.random.default_rng(cfg.bootstrap_seed)
    n_rows = len(per_sample)
    boot_rows: list[dict[str, float]] = []
    for b in range(cfg.n_bootstrap):
        idx = rng.integers(0, n_rows, size=n_rows)
        y_b = y[idx]
        qh = q_h[idx]
        qhn = q_hn[idx]
        qhn_perm = q_hn_perm[idx]
        h_b = h_raw[idx]
        boot_rows.append(
            {
                "bootstrap_id": b,
                "delta_L": _binary_log_loss(y_b, qh, eps=cfg.prob_epsilon)
                - _binary_log_loss(y_b, qhn, eps=cfg.prob_epsilon),
                "delta_L_perm": _binary_log_loss(y_b, qhn_perm, eps=cfg.prob_epsilon)
                - _binary_log_loss(y_b, qhn, eps=cfg.prob_epsilon),
                "delta_auroc": (
                    score_auroc_auprc(y_b, qhn)["auroc"] - score_auroc_auprc(y_b, h_b)["auroc"]
                    if len(np.unique(y_b)) >= 2
                    else float("nan")
                ),
                "L_H": _binary_log_loss(y_b, qh, eps=cfg.prob_epsilon),
                "L_HN": _binary_log_loss(y_b, qhn, eps=cfg.prob_epsilon),
                "L_HN_perm": _binary_log_loss(y_b, qhn_perm, eps=cfg.prob_epsilon),
            }
        )
    boot_df = pd.DataFrame(boot_rows)
    dl_mean, dl_lo, dl_hi = bootstrap_ci(boot_df["delta_L"].to_numpy(), ci_level=cfg.ci_level)
    dlp_mean, dlp_lo, dlp_hi = bootstrap_ci(boot_df["delta_L_perm"].to_numpy(), ci_level=cfg.ci_level)
    dau_mean, dau_lo, dau_hi = bootstrap_ci(boot_df["delta_auroc"].to_numpy(), ci_level=cfg.ci_level)

    stratified_df = stratified_n_auroc(per_sample, n_bins=cfg.n_entropy_bins)
    matched_df = matched_high_low_n_analysis(
        per_sample,
        n_h_bins=cfg.n_entropy_bins,
        n_bootstrap=cfg.n_bootstrap,
        bootstrap_seed=cfg.bootstrap_seed,
        ci_level=cfg.ci_level,
    )

    results = {
        "L_H": l_h,
        "L_HN": l_hn,
        "delta_L": l_h - l_hn,
        "delta_L_perm": l_hn_perm - l_hn,
        "auroc_H": auroc_h,
        "auroc_HN": auroc_hn,
        "delta_auroc": auroc_hn - auroc_h,
        "test_n": int(len(per_sample)),
        "test_error_rate": float(y.mean()),
        "delta_L_bootstrap_mean": dl_mean,
        "delta_L_ci_low": dl_lo,
        "delta_L_ci_high": dl_hi,
        "delta_L_ci_excludes_zero": bool(dl_lo > 0),
        "delta_L_perm_bootstrap_mean": dlp_mean,
        "delta_L_perm_ci_low": dlp_lo,
        "delta_L_perm_ci_high": dlp_hi,
        "delta_L_perm_ci_excludes_zero": bool(dlp_lo > 0),
        "delta_auroc_bootstrap_mean": dau_mean,
        "delta_auroc_ci_low": dau_lo,
        "delta_auroc_ci_high": dau_hi,
        "leakage": per_seed_runs[str(cfg.seeds[0])]["results"]["leakage"],
        "seed": "pooled",
    }
    gate = preregistered_decision(results, stratified_df, matched_df)
    results["decision_gate"] = gate

    return {
        "results": results,
        "bootstrap_df": boot_df,
        "stratified_df": stratified_df,
        "matched_df": matched_df,
        "per_sample_df": per_sample,
        "models": per_seed_runs[str(cfg.seeds[0])]["models"],
        "bin_edges": per_seed_runs[str(cfg.seeds[0])]["bin_edges"],
    }


def run_nro_final_experiment(cfg: NROFinalConfig) -> dict[str, Any]:
    cfg = cfg.resolve_paths()
    dirs = _ensure_output_dirs(cfg.output_dir)  # type: ignore[arg-type]

    cal_all = load_calibration_frames(cfg)
    test_all = load_test_frames(cfg)

    per_seed_runs: dict[str, Any] = {}
    seed_results_rows: list[dict[str, Any]] = []

    for seed in cfg.seeds:
        run = run_single_seed_analysis(cal_all, test_all, cfg, seed=seed)
        per_seed_runs[str(seed)] = run
        row = {k: v for k, v in run["results"].items() if k not in ("fit_meta", "leakage", "decision_gate")}
        row["seed"] = seed
        row["passed"] = run["results"]["decision_gate"]["passed"]
        seed_results_rows.append(row)

    pooled_run = aggregate_pooled_from_seed_runs(per_seed_runs, cfg)
    pooled_results = pooled_run["results"]
    gate = pooled_results["decision_gate"]

    results_df = pd.DataFrame(seed_results_rows)
    pooled_row = {k: v for k, v in pooled_results.items() if k not in ("fit_meta", "leakage", "decision_gate")}
    pooled_row["seed"] = "pooled"
    pooled_row["passed"] = gate["passed"]
    results_df = pd.concat([results_df, pd.DataFrame([pooled_row])], ignore_index=True)

    if cfg.save_artifacts:
        output_dir = cfg.output_dir
        assert output_dir is not None

        config_payload = {
            **asdict(cfg),
            "repo_root": str(cfg.repo_root),
            "source_experiment_dir": str(cfg.source_experiment_dir),
            "output_dir": str(cfg.output_dir),
            "seeds": list(cfg.seeds),
            "h_definition": "normalized_entropy = predictive_entropy / log(num_classes)",
            "n_definition": "memory_novelty = 1 - memory_agreement (frozen V5B MSA)",
            "e_definition": "error = 1[argmax p != y]",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "platform": platform.platform(),
        }
        (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

        checkpoint_paths = _checkpoint_paths(cfg)
        (dirs["audit"] / "checkpoint_paths.json").write_text(
            json.dumps(checkpoint_paths, indent=2), encoding="utf-8"
        )
        split_payload = {
            "calibration_sample_ids": cal_all["sample_id"].astype(str).tolist(),
            "test_sample_ids": test_all["sample_id"].astype(str).tolist(),
            "evaluation_domain": cfg.evaluation_domain,
        }
        (dirs["audit"] / "split_indices.json").write_text(json.dumps(split_payload, indent=2), encoding="utf-8")
        (dirs["audit"] / "leakage_assertions.json").write_text(
            json.dumps(pooled_run["results"]["leakage"], indent=2), encoding="utf-8"
        )

        results_df.to_csv(output_dir / "results.csv", index=False)
        pooled_run["per_sample_df"].to_csv(output_dir / "per_sample.csv", index=False)
        pooled_run["bootstrap_df"].to_csv(output_dir / "bootstrap_results.csv", index=False)
        pooled_run["stratified_df"].to_csv(output_dir / "stratified_results.csv", index=False)
        pooled_run["matched_df"].to_csv(output_dir / "matched_results.csv", index=False)
        (output_dir / "decision_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

        if cfg.show_plots:
            fig_dir = dirs["figures"]
            plot_failure_probability_curves(
                pooled_run["per_sample_df"],
                pooled_run["models"]["H"],
                pooled_run["models"]["HN"],
                fig_dir / "failure_probability_vs_h_hn.png",
            )
            plot_delta_logloss_bootstrap(
                pooled_run["bootstrap_df"],
                fig_dir / "delta_logloss_bootstrap.png",
                ci_level=cfg.ci_level,
            )
            plot_h_n_scatter(pooled_run["per_sample_df"], fig_dir / "h_vs_n_scatter.png")
            plot_stratified_auroc(pooled_run["stratified_df"], fig_dir / "stratified_n_auroc.png")
            plot_matched_failure_rates(pooled_run["matched_df"], fig_dir / "matched_failure_rates.png")

        readme = _generate_readme(cfg, pooled_results, gate, results_df)
        (output_dir / "README.md").write_text(readme, encoding="utf-8")

    return {
        "config": cfg,
        "results_df": results_df,
        "pooled_results": pooled_results,
        "decision_gate": gate,
        "per_seed_runs": per_seed_runs,
        "pooled_run": pooled_run,
        "output_dir": cfg.output_dir,
    }


def _generate_readme(
    cfg: NROFinalConfig,
    results: dict[str, Any],
    gate: dict[str, Any],
    results_df: pd.DataFrame,
) -> str:
    lines = [
        "# Final NRO Experiment — DermaMNIST Conditional Information",
        "",
        "## Question",
        "",
        "Does optimization-history novelty N contain sample-specific information about",
        "prediction failure that is not recoverable from predictive entropy H alone?",
        "",
        f"**I(E; N | H) > 0?** → **{'YES' if gate['answers_I_E_N_given_H_positive'] else 'NO'}**",
        "",
        f"**Pre-registered PASS:** **{gate['passed']}**",
        "",
        "## Primary results (pooled external test)",
        "",
        f"- ΔL = L_H − L_HN = **{results['delta_L']:.6f}**",
        f"- 95% CI: [{results['delta_L_ci_low']:.6f}, {results['delta_L_ci_high']:.6f}]",
        f"- ΔL_perm = L_HN,perm − L_HN = **{results['delta_L_perm']:.6f}**",
        f"- 95% CI: [{results['delta_L_perm_ci_low']:.6f}, {results['delta_L_perm_ci_high']:.6f}]",
        f"- AUROC H = {results['auroc_H']:.4f}, AUROC H+N = {results['auroc_HN']:.4f}",
        "",
        "## Per-seed summary",
        "",
        "```",
        results_df.to_string(index=False),
        "```",
        "",
        "## Interpretation",
        "",
        _interpretation_text(gate),
        "",
        "## Artifacts",
        "",
        "- `results.csv`, `per_sample.csv`, `bootstrap_results.csv`",
        "- `stratified_results.csv`, `matched_results.csv`",
        "- `config.json`, `decision_gate.json`, `figures/`",
    ]
    return "\n".join(lines)


def _interpretation_text(gate: dict[str, Any]) -> str:
    if gate["passed"]:
        return (
            "PASS: N reduces failure log-loss beyond H on external DermaMNIST-E, and the gain "
            "exceeds an H-conditioned N-permutation null. This supports **conditional information** "
            "I(E; N | H) > 0. It does **not** claim that N fully explains model failure or replaces H."
        )
    return (
        "FAIL: Pre-registered criteria not met. Cannot claim I(E; N | H) > 0 under this protocol. "
        "Does not invalidate prior AUROC-based clinical failure results — this is a stricter "
        "log-loss / permutation test."
    )


def print_final_verdict(results: dict[str, Any]) -> None:
    gate = results["decision_gate"]
    pooled = results["pooled_results"]
    print("=" * 72)
    print("FINAL NRO EXPERIMENT — PRE-REGISTERED VERDICT")
    print("=" * 72)
    print(f"  I(E; N | H) > 0 supported?     {gate['answers_I_E_N_given_H_positive']}")
    print(f"  Pre-registered PASS:           {gate['passed']}")
    print("-" * 72)
    print("  Primary:")
    print(f"    delta_L > 0:                 {gate['primary_delta_L_positive']}  ({pooled['delta_L']:.6f})")
    print(f"    CI(delta_L) excludes 0:      {gate['primary_delta_L_ci_excludes_zero']}")
    print(f"      95% CI: [{pooled['delta_L_ci_low']:.6f}, {pooled['delta_L_ci_high']:.6f}]")
    print("  Permutation control:")
    print(f"    delta_L_perm > 0:            {gate['permutation_delta_L_perm_positive']}  ({pooled['delta_L_perm']:.6f})")
    print(f"    CI(delta_L_perm) excludes 0: {gate['permutation_delta_L_perm_ci_excludes_zero']}")
    print(f"      95% CI: [{pooled['delta_L_perm_ci_low']:.6f}, {pooled['delta_L_perm_ci_high']:.6f}]")
    print("  Secondary (report only):")
    print(f"    AUROC(H+N) > AUROC(H):       {gate['secondary_auroc_improves']}")
    print(f"    Majority strata AUROC(N)>0.5: {gate['secondary_strata_majority_auroc_n_above_half']}")
    print(f"    Matched high-N > low-N:      {gate['secondary_matched_high_n_higher_failure_rate']}")
    print("=" * 72)
    print(_interpretation_text(gate))
    print("=" * 72)
