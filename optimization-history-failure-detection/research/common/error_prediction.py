"""Phase 4 — error prediction from predictive entropy and MSA signals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SINGLE_FEATURES: dict[str, str] = {
    "predictive_entropy": "predictive_entropy",
    "memory_novelty": "memory_novelty",
    "memory_disagreement": "memory_disagreement",
    "msa_entropy": "msa_entropy",
}

COMBINED_FEATURE_COLUMNS = list(SINGLE_FEATURES.values())
DEFAULT_BOOTSTRAP_SAMPLES = 500
DEFAULT_AUROC_MARGIN = 0.005


def add_error_target(df: pd.DataFrame) -> pd.DataFrame:
    """E_i(t) = 1[ŷ_i(t) ≠ y_i]."""
    out = df.copy()
    out["error"] = (1 - out["correct"].astype(int)).astype(int)
    return out


def _valid_mask(df: pd.DataFrame, columns: Iterable[str]) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)
    for col in columns:
        mask &= df[col].notna().to_numpy()
    return mask


def score_auroc_auprc(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
    }


def evaluate_single_feature(
    df: pd.DataFrame,
    feature: str,
    *,
    split: str | None = None,
) -> dict[str, float]:
    subset = df if split is None else df[df["split"] == split]
    mask = _valid_mask(subset, ["error", feature])
    if not mask.any():
        return {"auroc": float("nan"), "auprc": float("nan"), "n": 0}
    metrics = score_auroc_auprc(subset.loc[mask, "error"].to_numpy(), subset.loc[mask, feature].to_numpy())
    metrics["n"] = int(mask.sum())
    return metrics


def resolve_features(features: dict[str, str] | None) -> dict[str, str]:
    return dict(features) if features is not None else dict(SINGLE_FEATURES)


def evaluate_all_single_features(
    df: pd.DataFrame,
    *,
    splits: tuple[str, ...] = ("val", "test"),
    features: dict[str, str] | None = None,
) -> pd.DataFrame:
    feature_map = resolve_features(features)
    rows: list[dict[str, Any]] = []
    for split in splits:
        for name, column in feature_map.items():
            metrics = evaluate_single_feature(df, column, split=split)
            rows.append(
                {
                    "split": split,
                    "feature": name,
                    "column": column,
                    "auroc": metrics["auroc"],
                    "auprc": metrics["auprc"],
                    "n": metrics["n"],
                }
            )
    return pd.DataFrame(rows)


def fit_failure_scorer(
    df: pd.DataFrame,
    feature_columns: list[str],
    *,
    error_col: str = "error",
    max_iter: int = 2000,
) -> Pipeline:
    """Standardized failure scorer: StandardScaler + logistic regression."""
    mask = _valid_mask(df, [error_col, *feature_columns])
    if not mask.any():
        raise ValueError("No valid rows available to fit logistic scorer.")
    x = df.loc[mask, feature_columns].to_numpy(dtype=float)
    y = df.loc[mask, error_col].to_numpy(dtype=int)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=max_iter, random_state=0)),
        ]
    )
    model.fit(x, y)
    return model


def _fit_logistic_scorer(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> Pipeline:
    return fit_failure_scorer(df, feature_columns, max_iter=1000)


def pipeline_logistic_coefficients(model: Pipeline, feature_names: list[str]) -> dict[str, float]:
    clf = model.named_steps["clf"]
    out: dict[str, float] = {"intercept": float(clf.intercept_[0])}
    for name, coef in zip(feature_names, clf.coef_.ravel()):
        out[name] = float(coef)
    return out


def predict_error_probability(model: Pipeline, df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    mask = _valid_mask(df, feature_columns)
    scores = np.full(len(df), np.nan, dtype=float)
    if not mask.any():
        return scores
    scores[mask] = model.predict_proba(df.loc[mask, feature_columns].to_numpy(dtype=float))[:, 1]
    return scores


def evaluate_combined_models(
    df: pd.DataFrame,
    *,
    train_split: str = "val",
    eval_splits: tuple[str, ...] = ("val", "test"),
    combined_features: list[str] | None = None,
) -> pd.DataFrame:
    combined_cols = list(combined_features or COMBINED_FEATURE_COLUMNS)
    train_df = df[df["split"] == train_split]
    entropy_model = _fit_logistic_scorer(train_df, ["predictive_entropy"])
    combined_model = _fit_logistic_scorer(train_df, combined_cols)

    rows: list[dict[str, Any]] = []
    for split in eval_splits:
        eval_df = df[df["split"] == split].copy()
        for model_name, model, cols in (
            ("entropy_only", entropy_model, ["predictive_entropy"]),
            ("combined", combined_model, combined_cols),
        ):
            scores = predict_error_probability(model, eval_df, cols)
            mask = np.isfinite(scores)
            metrics = score_auroc_auprc(eval_df.loc[mask, "error"].to_numpy(), scores[mask])
            rows.append(
                {
                    "split": split,
                    "model": model_name,
                    "auroc": metrics["auroc"],
                    "auprc": metrics["auprc"],
                    "n": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_auroc_delta(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap AUROC(score_b) - AUROC(score_a) on paired rows."""
    y_true = np.asarray(y_true, dtype=int)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    n = len(y_true)
    if n == 0 or len(np.unique(y_true)) < 2:
        return {
            "delta_mean": float("nan"),
            "delta_ci_low": float("nan"),
            "delta_ci_high": float("nan"),
            "p_value": float("nan"),
        }

    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        auroc_a = roc_auc_score(y_b, score_a[idx])
        auroc_b = roc_auc_score(y_b, score_b[idx])
        deltas.append(auroc_b - auroc_a)

    if not deltas:
        return {
            "delta_mean": float("nan"),
            "delta_ci_low": float("nan"),
            "delta_ci_high": float("nan"),
            "p_value": float("nan"),
        }

    deltas_arr = np.asarray(deltas, dtype=float)
    return {
        "delta_mean": float(np.mean(deltas_arr)),
        "delta_ci_low": float(np.percentile(deltas_arr, 2.5)),
        "delta_ci_high": float(np.percentile(deltas_arr, 97.5)),
        "p_value": float(np.mean(deltas_arr <= 0.0)),
    }


def future_error_correlation(
    df: pd.DataFrame,
    feature_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Correlate uncertainty at epoch t with error at epoch t+1 within each sample."""
    feature_columns = list(feature_columns or SINGLE_FEATURES.values())
    rows: list[dict[str, Any]] = []
    grouped = df.sort_values(["sample_id", "epoch"]).groupby("sample_id", sort=False)
    for feature in feature_columns:
        current_vals: list[float] = []
        future_errors: list[int] = []
        for _, g in grouped:
            if len(g) < 2:
                continue
            g = g.sort_values("epoch")
            current = g[feature].to_numpy(dtype=float)
            future = g["error"].shift(-1).to_numpy(dtype=float)
            valid = np.isfinite(current) & np.isfinite(future)
            current_vals.extend(current[valid].tolist())
            future_errors.extend(future[valid].astype(int).tolist())
        if len(current_vals) < 2:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(np.asarray(current_vals), np.asarray(future_errors))[0, 1])
        rows.append({"feature": feature, "pearson_r_future_error": corr, "n_pairs": len(current_vals)})
    return pd.DataFrame(rows)


def binned_error_probability(
    df: pd.DataFrame,
    score_col: str,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    subset = df[[score_col, "error"]].dropna()
    if subset.empty:
        return pd.DataFrame(columns=["bin", "score_mid", "error_rate", "count"])
    scores = subset[score_col].to_numpy(dtype=float)
    errors = subset["error"].to_numpy(dtype=int)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(scores, quantiles))
    if len(edges) < 3:
        edges = np.linspace(scores.min(), scores.max(), min(n_bins + 1, len(scores) + 1))
    bin_idx = np.digitize(scores, edges[1:-1], right=False)
    rows: list[dict[str, Any]] = []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": int(b),
                "score_low": float(edges[b]),
                "score_high": float(edges[b + 1]),
                "score_mid": float(0.5 * (edges[b] + edges[b + 1])),
                "error_rate": float(errors[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_error_probability_vs_uncertainty(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    features: dict[str, str] | None = None,
    split: str = "test",
    n_bins: int = 10,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features = features or SINGLE_FEATURES
    subset = df[df["split"] == split] if split else df
    paths: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, column in features.items():
        bins_df = binned_error_probability(subset, column, n_bins=n_bins)
        if bins_df.empty:
            continue
        ax.plot(bins_df["score_mid"], bins_df["error_rate"], marker="o", label=label)
    ax.set_xlabel("uncertainty score")
    ax.set_ylabel("error probability")
    ax.set_title(f"Error probability vs uncertainty ({split})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / f"error_probability_vs_uncertainty_{split}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    paths["combined"] = path

    for label, column in features.items():
        bins_df = binned_error_probability(subset, column, n_bins=n_bins)
        if bins_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(bins_df["score_mid"], bins_df["error_rate"], marker="o", color="steelblue")
        ax.set_xlabel(column)
        ax.set_ylabel("error probability")
        ax.set_title(f"{label} ({split})")
        fig.tight_layout()
        feature_path = out_dir / f"error_probability_vs_{label}_{split}.png"
        fig.savefig(feature_path, dpi=120)
        plt.close(fig)
        paths[label] = feature_path
    return paths


def decision_gate(
    single_metrics: pd.DataFrame,
    combined_metrics: pd.DataFrame,
    *,
    eval_split: str = "test",
    auroc_margin: float = DEFAULT_AUROC_MARGIN,
    msa_feature_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return PASS only if MSA signals add predictive information beyond entropy."""
    test_single = single_metrics[single_metrics["split"] == eval_split]
    entropy_row = test_single[test_single["feature"] == "predictive_entropy"].iloc[0]
    entropy_auroc = float(entropy_row["auroc"])

    if msa_feature_names is None:
        msa_features = [name for name in SINGLE_FEATURES if name != "predictive_entropy"]
    else:
        msa_features = [name for name in msa_feature_names if name != "predictive_entropy"]
    msa_rows = test_single[test_single["feature"].isin(msa_features)]
    best_msa = msa_rows.loc[msa_rows["auroc"].idxmax()] if len(msa_rows) else None
    best_msa_name = str(best_msa["feature"]) if best_msa is not None else ""
    best_msa_auroc = float(best_msa["auroc"]) if best_msa is not None else float("nan")

    test_combined = combined_metrics[combined_metrics["split"] == eval_split]
    entropy_model = test_combined[test_combined["model"] == "entropy_only"].iloc[0]
    combined_model = test_combined[test_combined["model"] == "combined"].iloc[0]
    combined_auroc = float(combined_model["auroc"])
    delta = combined_auroc - float(entropy_model["auroc"])

    msa_beats_entropy = bool(np.isfinite(best_msa_auroc) and best_msa_auroc > entropy_auroc + auroc_margin)
    combined_beats_entropy = bool(np.isfinite(delta) and delta > auroc_margin)
    passed = msa_beats_entropy or combined_beats_entropy

    if passed:
        decision = (
            "MSA features add predictive information beyond predictive entropy. "
            "Proceed to Phase 5 (distribution-shift test)."
        )
        next_phase = "Phase 5 — distribution-shift / OOD detection benchmark."
    else:
        decision = (
            "MSA features do not add predictive information beyond predictive entropy. "
            "STOP this research branch."
        )
        next_phase = "None — failed Phase 4 decision gate."

    return {
        "passed": passed,
        "entropy_auroc": entropy_auroc,
        "best_msa_feature": best_msa_name,
        "best_msa_auroc": best_msa_auroc,
        "combined_auroc": combined_auroc,
        "entropy_model_auroc": float(entropy_model["auroc"]),
        "combined_delta_auroc": delta,
        "auroc_margin": auroc_margin,
        "msa_beats_entropy": msa_beats_entropy,
        "combined_beats_entropy": combined_beats_entropy,
        "decision": decision,
        "next_phase": next_phase,
    }


def write_phase4_readme(path: Path, gate: dict[str, Any], *, eval_split: str = "test") -> None:
    status = "PASS" if gate["passed"] else "FAIL"
    lines = [
        "# Phase 4 — Error prediction test",
        "",
        "## Hypothesis",
        "",
        "Memory-derived MSA signals (novelty, disagreement, MSA entropy) predict",
        "model errors better than predictive entropy alone.",
        "",
        "## Experiment",
        "",
        "- Target: E_i(t) = 1[ŷ_i(t) ≠ y_i]",
        "- Scorers: predictive entropy, MSA novelty, MSA disagreement, MSA entropy,",
        "  and a combined logistic model (entropy + MSA features)",
        f"- Primary evaluation split: `{eval_split}`",
        "- Metrics: AUROC, AUPRC, correlation with future error",
        "",
        "## Result",
        "",
        f"- Predictive entropy AUROC ({eval_split}): {gate['entropy_auroc']:.4f}",
        f"- Best single MSA feature: `{gate['best_msa_feature']}` "
        f"(AUROC={gate['best_msa_auroc']:.4f})",
        f"- Combined model AUROC ({eval_split}): {gate['combined_auroc']:.4f}",
        f"- ΔAUROC (combined − entropy-only): {gate['combined_delta_auroc']:.4f}",
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


def run_phase4_analysis(
    per_sample_df: pd.DataFrame,
    output_dir: Path,
    *,
    train_split: str = "val",
    eval_splits: tuple[str, ...] = ("val", "test"),
    bootstrap_split: str = "test",
    n_bins: int = 10,
    features: dict[str, str] | None = None,
    combined_features: list[str] | None = None,
    auroc_margin: float = DEFAULT_AUROC_MARGIN,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = 0,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    feature_map = resolve_features(features)
    combined_cols = list(combined_features or list(feature_map.values()))

    if save_artifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

    df = add_error_target(per_sample_df)
    single_metrics = evaluate_all_single_features(df, splits=eval_splits, features=feature_map)
    combined_metrics = evaluate_combined_models(
        df,
        train_split=train_split,
        eval_splits=eval_splits,
        combined_features=combined_cols,
    )
    future_corr = future_error_correlation(df, feature_columns=list(feature_map.values()))

    test_df = df[df["split"] == bootstrap_split].copy()
    entropy_model = _fit_logistic_scorer(df[df["split"] == train_split], ["predictive_entropy"])
    combined_model = _fit_logistic_scorer(df[df["split"] == train_split], combined_cols)
    entropy_scores = predict_error_probability(entropy_model, test_df, ["predictive_entropy"])
    combined_scores = predict_error_probability(combined_model, test_df, combined_cols)
    mask = np.isfinite(entropy_scores) & np.isfinite(combined_scores)
    bootstrap = bootstrap_auroc_delta(
        test_df.loc[mask, "error"].to_numpy(),
        entropy_scores[mask],
        combined_scores[mask],
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed,
    )

    gate = decision_gate(
        single_metrics,
        combined_metrics,
        eval_split=bootstrap_split,
        auroc_margin=auroc_margin,
        msa_feature_names=feature_map.keys(),
    )
    gate["bootstrap_delta_mean"] = bootstrap["delta_mean"]
    gate["bootstrap_delta_ci_low"] = bootstrap["delta_ci_low"]
    gate["bootstrap_delta_ci_high"] = bootstrap["delta_ci_high"]
    gate["bootstrap_p_value"] = bootstrap["p_value"]

    if save_artifacts:
        plot_error_probability_vs_uncertainty(
            df,
            plots_dir,
            features=feature_map,
            split=bootstrap_split,
            n_bins=n_bins,
        )

        per_sample_df.to_csv(output_dir / "per_sample.csv", index=False)
        single_metrics.to_csv(output_dir / "metrics.csv", index=False)
        combined_metrics.to_csv(output_dir / "metrics_combined.csv", index=False)
        future_corr.to_csv(output_dir / "future_error_correlation.csv", index=False)
        pd.DataFrame([gate]).to_csv(output_dir / "decision_gate.csv", index=False)

        config = {
            "phase": 4,
            "train_split": train_split,
            "eval_splits": list(eval_splits),
            "bootstrap_split": bootstrap_split,
            "n_bins": n_bins,
            "auroc_margin": auroc_margin,
            "n_bootstrap": n_bootstrap,
            "bootstrap_seed": bootstrap_seed,
            "single_features": feature_map,
            "combined_features": combined_cols,
            "decision_gate": gate,
        }
        (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        write_phase4_readme(output_dir / "README.md", gate, eval_split=bootstrap_split)

    return {
        "dataframe": df,
        "single_metrics": single_metrics,
        "combined_metrics": combined_metrics,
        "future_correlation": future_corr,
        "decision_gate": gate,
        "plots_dir": plots_dir,
        "output_dir": output_dir,
    }
