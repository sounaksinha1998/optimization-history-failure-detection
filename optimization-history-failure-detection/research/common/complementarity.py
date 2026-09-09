"""Memory novelty vs predictive entropy complementarity analysis."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from research.common.error_prediction import (
    _valid_mask,
    bootstrap_auroc_delta,
    predict_error_probability,
    score_auroc_auprc,
)

ThresholdMethod = Literal["median", "quantile", "fixed"]
QUADRANT_LABELS = ("A", "B", "C", "D")


@dataclass
class ComplementarityConfig:
    """Configuration for the memory-novelty complementarity experiment."""

    repo_root: Path | None = None
    input_csv: Path | None = None
    output_dir: Path | None = None
    run_phase3_if_missing: bool = False

    seeds: Sequence[int] | None = None
    epochs: Sequence[int] | None = None
    calibration_split: str = "val"
    test_split: str = "test"

    entropy_col: str = "predictive_entropy"
    novelty_col: str = "memory_novelty"

    threshold_method: ThresholdMethod = "median"
    threshold_quantile: float = 0.5
    h_high_fixed: float | None = None
    n_high_fixed: float | None = None

    n_entropy_bins: int = 4
    min_quadrant_samples: int = 30

    n_bootstrap: int = 1000
    bootstrap_seed: int = 0
    ci_level: float = 0.95

    auroc_margin: float = 0.005
    min_quadrant_b_minus_a: float = 0.02

    save_artifacts: bool = True
    show_plots: bool = True

    def resolve_paths(self, cwd: Path | None = None) -> ComplementarityConfig:
        root = self.repo_root
        if root is None:
            root = Path(cwd or Path.cwd()).resolve()
            if not (root / "research").exists() and (root.parent / "research").exists():
                root = root.parent
        self.repo_root = root
        if self.input_csv is None:
            self.input_csv = root / "research" / "phase3_signals" / "phase3_signals.csv"
        if self.output_dir is None:
            self.output_dir = root / "research" / "memory_novelty_complementarity"
        return self


def apply_row_filters(
    df: pd.DataFrame,
    *,
    seeds: Sequence[int] | None = None,
    epochs: Sequence[int] | None = None,
    splits: Sequence[str] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if seeds is not None and "seed" in out.columns:
        out = out[out["seed"].isin(seeds)]
    if epochs is not None and "epoch" in out.columns:
        out = out[out["epoch"].isin(epochs)]
    if splits is not None and "split" in out.columns:
        out = out[out["split"].isin(splits)]
    return out.reset_index(drop=True)


def build_per_example_table(
    df: pd.DataFrame,
    *,
    entropy_col: str = "predictive_entropy",
    novelty_col: str = "memory_novelty",
) -> pd.DataFrame:
    """Build H_i, N_i, E_i table from Phase 3 per-sample rows."""
    required = {"correct", entropy_col, novelty_col, "split", "sample_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out["H"] = out[entropy_col].astype(float)
    out["N"] = out[novelty_col].astype(float)
    out["error"] = (1 - pd.to_numeric(out["correct"], errors="coerce").fillna(0)).astype(int)
    return out


def fit_thresholds(
    cal_df: pd.DataFrame,
    *,
    threshold_method: ThresholdMethod = "median",
    threshold_quantile: float = 0.5,
    h_high_fixed: float | None = None,
    n_high_fixed: float | None = None,
) -> dict[str, float]:
    if threshold_method == "fixed":
        if h_high_fixed is None or n_high_fixed is None:
            raise ValueError("h_high_fixed and n_high_fixed required for threshold_method='fixed'")
        return {"H_high": float(h_high_fixed), "N_high": float(n_high_fixed)}

    if threshold_method == "median":
        q = 0.5
    elif threshold_method == "quantile":
        q = threshold_quantile
    else:
        raise ValueError(f"Unknown threshold_method: {threshold_method!r}")

    return {
        "H_high": float(cal_df["H"].quantile(q)),
        "N_high": float(cal_df["N"].quantile(q)),
    }


def assign_quadrant(
    df: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.Series:
    h_high = thresholds["H_high"]
    n_high = thresholds["N_high"]
    h_low = df["H"] <= h_high
    n_low = df["N"] <= n_high

    quadrant = pd.Series(index=df.index, dtype=object)
    quadrant[h_low & n_low] = "A"
    quadrant[h_low & ~n_low] = "B"
    quadrant[~h_low & n_low] = "C"
    quadrant[~h_low & ~n_low] = "D"
    return quadrant


def bootstrap_binomial_ci(
    errors: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
    ci_level: float = 0.95,
) -> dict[str, float]:
    errors = np.asarray(errors, dtype=int)
    n = len(errors)
    if n == 0:
        return {"error_rate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}

    rate = float(errors.mean())
    if n == 1:
        return {"error_rate": rate, "ci_low": rate, "ci_high": rate, "n": n}

    rng = np.random.default_rng(seed)
    boot_rates: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_rates.append(float(errors[idx].mean()))

    alpha = (1.0 - ci_level) / 2.0
    low = float(np.percentile(boot_rates, 100 * alpha))
    high = float(np.percentile(boot_rates, 100 * (1 - alpha)))
    return {"error_rate": rate, "ci_low": low, "ci_high": high, "n": n}


def quadrant_error_rates(
    test_df: pd.DataFrame,
    *,
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 0,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for q in QUADRANT_LABELS:
        subset = test_df[test_df["quadrant"] == q]
        stats = bootstrap_binomial_ci(
            subset["error"].to_numpy(),
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed + hash(q) % 10_000,
            ci_level=ci_level,
        )
        rows.append(
            {
                "quadrant": q,
                "label": _quadrant_label(q),
                "n": stats["n"],
                "error_rate": stats["error_rate"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
            }
        )
    return pd.DataFrame(rows)


def _quadrant_label(q: str) -> str:
    labels = {
        "A": "low H / low N",
        "B": "low H / high N",
        "C": "high H / low N",
        "D": "high H / high N",
    }
    return labels.get(q, q)


def compare_quadrants(
    quadrant_df: pd.DataFrame,
    q_a: str,
    q_b: str,
    test_df: pd.DataFrame,
    *,
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 0,
    ci_level: float = 0.95,
) -> dict[str, float]:
    """Bootstrap CI for P(E|q_b) - P(E|q_a)."""
    a_errors = test_df.loc[test_df["quadrant"] == q_a, "error"].to_numpy(dtype=int)
    b_errors = test_df.loc[test_df["quadrant"] == q_b, "error"].to_numpy(dtype=int)
    if len(a_errors) == 0 or len(b_errors) == 0:
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    rng = np.random.default_rng(bootstrap_seed)
    deltas: list[float] = []
    for _ in range(n_bootstrap):
        a_idx = rng.integers(0, len(a_errors), size=len(a_errors))
        b_idx = rng.integers(0, len(b_errors), size=len(b_errors))
        deltas.append(float(b_errors[b_idx].mean() - a_errors[a_idx].mean()))

    alpha = (1.0 - ci_level) / 2.0
    return {
        "quadrant_a": q_a,
        "quadrant_b": q_b,
        "rate_a": float(a_errors.mean()),
        "rate_b": float(b_errors.mean()),
        "delta": float(np.mean(deltas)),
        "ci_low": float(np.percentile(deltas, 100 * alpha)),
        "ci_high": float(np.percentile(deltas, 100 * (1 - alpha))),
    }


def _fit_logistic(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> Pipeline:
    mask = _valid_mask(df, ["error", *feature_columns])
    if not mask.any():
        raise ValueError("No valid rows to fit logistic model.")
    x = df.loc[mask, feature_columns].to_numpy(dtype=float)
    y = df.loc[mask, "error"].to_numpy(dtype=int)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=0)),
        ]
    )
    model.fit(x, y)
    return model


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    if len(y_true) == 0:
        return float("nan")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob >= bins[i]) & (y_prob < bins[i + 1] if i < n_bins - 1 else y_prob <= bins[i + 1])
        if not in_bin.any():
            continue
        acc = y_true[in_bin].mean()
        conf = y_prob[in_bin].mean()
        ece += in_bin.mean() * abs(acc - conf)
    return float(ece)


def evaluate_logistic_model(
    model: Pipeline,
    df: pd.DataFrame,
    feature_columns: list[str],
    *,
    model_name: str,
    split: str,
) -> dict[str, Any]:
    probs = predict_error_probability(model, df, feature_columns)
    mask = np.isfinite(probs)
    y = df.loc[mask, "error"].to_numpy(dtype=int)
    p = probs[mask]
    metrics = score_auroc_auprc(y, p)
    return {
        "model": model_name,
        "split": split,
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "n": int(mask.sum()),
        "coefficients": _extract_coefficients(model, feature_columns),
    }


def _extract_coefficients(model: Pipeline, feature_columns: list[str]) -> dict[str, float]:
    clf = model.named_steps["clf"]
    coefs = clf.coef_.ravel()
    out = {"intercept": float(clf.intercept_[0])}
    for name, coef in zip(feature_columns, coefs):
        out[name] = float(coef)
    return out


def fit_and_evaluate_logistic_models(
    df: pd.DataFrame,
    *,
    calibration_split: str,
    test_split: str,
    entropy_col: str = "predictive_entropy",
    novelty_col: str = "memory_novelty",
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    cal = df[df["split"] == calibration_split].copy()
    test = df[df["split"] == test_split].copy()
    cal["HN"] = cal["H"] * cal["N"]
    test["HN"] = test["H"] * test["N"]

    specs = {
        "H_only": ["H"],
        "H_plus_N": ["H", "N"],
        "H_plus_N_plus_HN": ["H", "N", "HN"],
    }
    models: dict[str, Pipeline] = {}
    rows: list[dict[str, Any]] = []
    for name, cols in specs.items():
        model = _fit_logistic(cal, cols)
        models[name] = model
        result = evaluate_logistic_model(model, test, cols, model_name=name, split=test_split)
        result["coefficients_json"] = json.dumps(result.pop("coefficients"))
        rows.append(result)

    return pd.DataFrame(rows), models


def stratified_novelty_auroc(
    test_df: pd.DataFrame,
    *,
    n_bins: int = 4,
) -> pd.DataFrame:
    df = test_df.dropna(subset=["H", "N", "error"]).copy()
    if df.empty:
        return pd.DataFrame()

    df["entropy_bin"] = pd.qcut(df["H"], q=n_bins, duplicates="drop")
    rows: list[dict[str, Any]] = []
    for bin_label, group in df.groupby("entropy_bin", observed=True):
        y = group["error"].to_numpy(dtype=int)
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
            }
        )
    return pd.DataFrame(rows)


def plot_reliability_map_2d(test_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    correct = test_df[test_df["error"] == 0]
    wrong = test_df[test_df["error"] == 1]
    ax.scatter(correct["H"], correct["N"], alpha=0.25, s=8, c="steelblue", label="correct")
    ax.scatter(wrong["H"], wrong["N"], alpha=0.35, s=8, c="crimson", label="error")
    ax.set_xlabel("predictive entropy (H)")
    ax.set_ylabel("memory novelty (N)")
    ax.set_title("2D reliability map")
    ax.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_quadrant_error_rates(quadrant_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(quadrant_df))
    rates = quadrant_df["error_rate"].to_numpy()
    yerr = np.vstack(
        [
            rates - quadrant_df["ci_low"].to_numpy(),
            quadrant_df["ci_high"].to_numpy() - rates,
        ]
    )
    colors = ["steelblue", "crimson", "darkorange", "purple"]
    ax.bar(x, rates, yerr=yerr, capsize=4, color=colors[: len(quadrant_df)], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['quadrant']}\n{r['label']}" for _, r in quadrant_df.iterrows()],
        fontsize=9,
    )
    ax.set_ylabel("error rate")
    ax.set_title("Quadrant error rates (B vs A is critical)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_error_probability_curves(
    models: dict[str, Pipeline],
    test_df: pd.DataFrame,
    out_path: Path,
    *,
    n_bins: int = 10,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    test = test_df.copy()
    test["HN"] = test["H"] * test["N"]
    specs = {"H_only": ["H"], "H_plus_N": ["H", "N"]}
    for name, cols in specs.items():
        probs = predict_error_probability(models[name], test, cols)
        mask = np.isfinite(probs)
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_rates = []
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            in_bin = mask & (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
            if not in_bin.any():
                continue
            bin_centers.append((lo + hi) / 2)
            bin_rates.append(float(test.loc[in_bin, "error"].mean()))
        ax.plot(bin_centers, bin_rates, marker="o", label=name.replace("_", " "))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect calibration")
    ax.set_xlabel("predicted P(error)")
    ax.set_ylabel("observed error rate")
    ax.set_title("Error probability curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_novelty_within_entropy_bins(stratified_df: pd.DataFrame, test_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(stratified_df), figsize=(4 * max(len(stratified_df), 1), 4), sharey=True)
    if len(stratified_df) == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, stratified_df.iterrows()):
        bin_mask = (test_df["H"] >= row["h_min"]) & (test_df["H"] <= row["h_max"])
        subset = test_df.loc[bin_mask].dropna(subset=["N", "error"])
        if len(subset) < 5:
            ax.set_title(f"bin {row.name}\n(n={len(subset)})")
            continue
        n_bins = min(8, max(3, len(subset) // 20))
        try:
            subset = subset.copy()
            subset["n_bin"] = pd.qcut(subset["N"], q=n_bins, duplicates="drop")
            grouped = subset.groupby("n_bin", observed=True)["error"].mean()
            ax.plot(range(len(grouped)), grouped.to_numpy(), marker="o")
            ax.set_title(f"H ∈ [{row['h_min']:.2f}, {row['h_max']:.2f}]\nAUROC(N)={row['auroc_n']:.3f}")
        except ValueError:
            ax.set_title(f"bin {row.name}\n(insufficient variation)")
        ax.set_xlabel("novelty bin")
    axes[0].set_ylabel("P(error)")
    fig.suptitle("Novelty → P(error) within entropy bins")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def decision_gate(
    quadrant_df: pd.DataFrame,
    b_vs_a: dict[str, float],
    logistic_df: pd.DataFrame,
    stratified_df: pd.DataFrame,
    bootstrap_delta: dict[str, float],
    *,
    auroc_margin: float = 0.005,
    min_quadrant_b_minus_a: float = 0.02,
) -> dict[str, Any]:
    row_a = quadrant_df[quadrant_df["quadrant"] == "A"].iloc[0]
    row_b = quadrant_df[quadrant_df["quadrant"] == "B"].iloc[0]

    h_only = logistic_df[logistic_df["model"] == "H_only"].iloc[0]
    h_plus_n = logistic_df[logistic_df["model"] == "H_plus_N"].iloc[0]
    h_plus_n_hn = logistic_df[logistic_df["model"] == "H_plus_N_plus_HN"].iloc[0]

    beta_n = float("nan")
    beta_hn = float("nan")
    if "coefficients_json" in h_plus_n.index:
        try:
            coefs = json.loads(h_plus_n["coefficients_json"])
            beta_n = float(coefs.get("N", float("nan")))
        except (json.JSONDecodeError, TypeError):
            pass
    if "coefficients_json" in h_plus_n_hn.index:
        try:
            coefs_hn = json.loads(h_plus_n_hn["coefficients_json"])
            beta_hn = float(coefs_hn.get("HN", float("nan")))
        except (json.JSONDecodeError, TypeError):
            pass

    b_gt_a = (
        b_vs_a["delta"] > min_quadrant_b_minus_a
        and b_vs_a["ci_low"] > 0
    )
    auroc_improves = (
        bootstrap_delta["delta_mean"] > auroc_margin
        and bootstrap_delta["delta_ci_low"] > 0
    )
    beta2_nonzero = np.isfinite(beta_n) and abs(beta_n) > 1e-6
    stratified_positive = (
        len(stratified_df) > 0
        and (stratified_df["auroc_n"] > 0.5).sum() >= max(1, len(stratified_df) // 2)
    )

    passed = b_gt_a and auroc_improves and beta2_nonzero and stratified_positive

    if passed:
        decision = (
            "Memory novelty provides information beyond predictive entropy. "
            "Proceed to Phase 5 (distribution-shift test)."
        )
        next_phase = "Phase 5 — distribution-shift / OOD benchmark."
    else:
        decision = (
            "Memory novelty does not convincingly add information beyond predictive entropy. "
            "STOP this research branch."
        )
        next_phase = "None — failed complementarity decision gate."

    return {
        "passed": passed,
        "b_error_rate": float(row_b["error_rate"]),
        "a_error_rate": float(row_a["error_rate"]),
        "b_minus_a_delta": b_vs_a["delta"],
        "b_minus_a_ci_low": b_vs_a["ci_low"],
        "b_minus_a_ci_high": b_vs_a["ci_high"],
        "auroc_h_only": float(h_only["auroc"]),
        "auroc_h_plus_n": float(h_plus_n["auroc"]),
        "delta_auroc": bootstrap_delta["delta_mean"],
        "delta_auroc_ci_low": bootstrap_delta["delta_ci_low"],
        "delta_auroc_ci_high": bootstrap_delta["delta_ci_high"],
        "beta_n": float(beta_n),
        "beta_hn": float(beta_hn),
        "stratified_bins_auroc_gt_half": int((stratified_df["auroc_n"] > 0.5).sum())
        if len(stratified_df)
        else 0,
        "stratified_bins_total": len(stratified_df),
        "checks": {
            "b_gt_a": b_gt_a,
            "auroc_improves": auroc_improves,
            "beta2_nonzero": beta2_nonzero,
            "stratified_positive": stratified_positive,
        },
        "decision": decision,
        "next_phase": next_phase,
    }


def write_complementarity_readme(path: Path, gate: dict[str, Any], cfg: ComplementarityConfig) -> None:
    status = "PASS" if gate["passed"] else "FAIL"
    lines = [
        "# Memory Novelty Complementarity Experiment",
        "",
        "## Hypothesis",
        "",
        "Memory novelty N detects errors that predictive entropy H alone misses.",
        "",
        "## Experiment",
        "",
        f"- Input: `{cfg.input_csv}`",
        f"- Calibration split: `{cfg.calibration_split}`",
        f"- Test split: `{cfg.test_split}`",
        f"- Threshold method: `{cfg.threshold_method}`",
        "",
        "## Result",
        "",
        f"- P(E|B) = {gate['b_error_rate']:.4f}, P(E|A) = {gate['a_error_rate']:.4f}",
        f"- Δ P(E|B) - P(E|A) = {gate['b_minus_a_delta']:.4f} "
        f"[{gate['b_minus_a_ci_low']:.4f}, {gate['b_minus_a_ci_high']:.4f}]",
        f"- AUROC H-only = {gate['auroc_h_only']:.4f}, H+N = {gate['auroc_h_plus_n']:.4f}",
        f"- ΔAUROC = {gate['delta_auroc']:.4f} "
        f"[{gate['delta_auroc_ci_low']:.4f}, {gate['delta_auroc_ci_high']:.4f}]",
        f"- β_N = {gate['beta_n']:.4f}, β_HN = {gate['beta_hn']:.4f}",
        f"- Decision gate: **{status}**",
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


def run_complementarity_analysis(
    per_sample_df: pd.DataFrame,
    cfg: ComplementarityConfig,
) -> dict[str, Any]:
    cfg = cfg.resolve_paths()
    output_dir = Path(cfg.output_dir)
    plots_dir = output_dir / "plots"
    if cfg.save_artifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

    per_example = build_per_example_table(
        per_sample_df,
        entropy_col=cfg.entropy_col,
        novelty_col=cfg.novelty_col,
    )

    cal_df = per_example[per_example["split"] == cfg.calibration_split].dropna(subset=["H", "N"])
    test_df = per_example[per_example["split"] == cfg.test_split].dropna(subset=["H", "N"]).copy()

    thresholds = fit_thresholds(
        cal_df,
        threshold_method=cfg.threshold_method,
        threshold_quantile=cfg.threshold_quantile,
        h_high_fixed=cfg.h_high_fixed,
        n_high_fixed=cfg.n_high_fixed,
    )
    test_df["quadrant"] = assign_quadrant(test_df, thresholds)
    test_df["H_high"] = thresholds["H_high"]
    test_df["N_high"] = thresholds["N_high"]

    quadrant_df = quadrant_error_rates(
        test_df,
        n_bootstrap=cfg.n_bootstrap,
        bootstrap_seed=cfg.bootstrap_seed,
        ci_level=cfg.ci_level,
    )
    b_vs_a = compare_quadrants(
        quadrant_df, "A", "B", test_df,
        n_bootstrap=cfg.n_bootstrap,
        bootstrap_seed=cfg.bootstrap_seed,
        ci_level=cfg.ci_level,
    )
    c_vs_a_reverse = compare_quadrants(
        quadrant_df, "A", "C", test_df,
        n_bootstrap=cfg.n_bootstrap,
        bootstrap_seed=cfg.bootstrap_seed + 1,
        ci_level=cfg.ci_level,
    )

    logistic_df, models = fit_and_evaluate_logistic_models(
        per_example,
        calibration_split=cfg.calibration_split,
        test_split=cfg.test_split,
    )

    test_cal = per_example[per_example["split"] == cfg.test_split].copy()
    test_cal["HN"] = test_cal["H"] * test_cal["N"]
    h_probs = predict_error_probability(models["H_only"], test_cal, ["H"])
    hn_probs = predict_error_probability(models["H_plus_N"], test_cal, ["H", "N"])
    mask = np.isfinite(h_probs) & np.isfinite(hn_probs)
    bootstrap_delta = bootstrap_auroc_delta(
        test_cal.loc[mask, "error"].to_numpy(),
        h_probs[mask],
        hn_probs[mask],
        n_bootstrap=cfg.n_bootstrap,
        seed=cfg.bootstrap_seed,
    )

    stratified_df = stratified_novelty_auroc(test_df, n_bins=cfg.n_entropy_bins)

    gate = decision_gate(
        quadrant_df,
        b_vs_a,
        logistic_df,
        stratified_df,
        bootstrap_delta,
        auroc_margin=cfg.auroc_margin,
        min_quadrant_b_minus_a=cfg.min_quadrant_b_minus_a,
    )

    if cfg.save_artifacts:
        per_example_out = per_example.copy()
        if "quadrant" in test_df.columns:
            per_example_out = per_example_out.merge(
                test_df[["sample_id", "epoch", "seed", "quadrant", "H_high", "N_high"]],
                on=["sample_id", "epoch", "seed"],
                how="left",
            )
        per_example_out.to_csv(output_dir / "per_example.csv", index=False)
        quadrant_df.to_csv(output_dir / "quadrant_error_rates.csv", index=False)
        pd.DataFrame([b_vs_a]).to_csv(output_dir / "b_vs_a_comparison.csv", index=False)
        pd.DataFrame([c_vs_a_reverse]).to_csv(output_dir / "reverse_comparison.csv", index=False)
        logistic_df.to_csv(output_dir / "logistic_models.csv", index=False)
        stratified_df.to_csv(output_dir / "stratified_auroc.csv", index=False)
        pd.DataFrame([gate]).to_csv(output_dir / "decision_gate.csv", index=False)

        plot_reliability_map_2d(test_df, plots_dir / "reliability_map_2d.png")
        plot_quadrant_error_rates(quadrant_df, plots_dir / "quadrant_error_rates.png")
        plot_error_probability_curves(models, test_df, plots_dir / "error_probability_curves.png")
        if len(stratified_df):
            plot_novelty_within_entropy_bins(stratified_df, test_df, plots_dir / "novelty_within_entropy_bins.png")

        config_out = {
            **asdict(cfg),
            "thresholds": thresholds,
            "decision_gate": gate,
        }
        for key in ("repo_root", "input_csv", "output_dir"):
            if key in config_out and config_out[key] is not None:
                config_out[key] = str(config_out[key])
        (output_dir / "config.json").write_text(json.dumps(config_out, indent=2, default=str), encoding="utf-8")
        write_complementarity_readme(output_dir / "README.md", gate, cfg)

    return {
        "per_example": per_example,
        "test_df": test_df,
        "thresholds": thresholds,
        "quadrant_df": quadrant_df,
        "b_vs_a": b_vs_a,
        "reverse_comparison": c_vs_a_reverse,
        "logistic_df": logistic_df,
        "stratified_df": stratified_df,
        "bootstrap_delta": bootstrap_delta,
        "decision_gate": gate,
        "models": models,
        "output_dir": output_dir,
    }
