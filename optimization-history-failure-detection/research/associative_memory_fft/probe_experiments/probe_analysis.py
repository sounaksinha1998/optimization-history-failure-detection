"""Leakage-safe probe analysis and publication figure generation.

Select X_best by calibration AUROC only; test is held out for final metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from research.common.clinical_failure import deferral_metrics_at_coverage, risk_coverage_curve

REPO_ROOT = Path(__file__).resolve().parents[3]
PROBE_DIR = Path(__file__).resolve().parent
FEATURE_CACHE = PROBE_DIR / "feature_cache"
PUB_IMAGES = REPO_ROOT / "publication" / "images"

SEEDS = [42, 123, 456]
TEST_SAMPLE_SIZE = 2000
RANDOM_SEED = 42
LOGISTIC_C = 1.0
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260908
NUM_Z_DIMS = 7
MAG_COLS = [f"z{j}_magnitude" for j in range(1, 5)]
COVERAGES = [0.70, 0.80, 0.90, 0.95]
DEFERRAL_RATES = [0.05, 0.10, 0.20, 0.30]


def level_cols(j: int) -> list[str]:
    return [f"z{j}_d{d}" for d in range(NUM_Z_DIMS)]


def cumulative_cols(max_level: int) -> list[str]:
    cols: list[str] = []
    for j in range(1, max_level + 1):
        cols.extend(level_cols(j))
    return cols


FULL_COLS = cumulative_cols(4)


def subsample_test_df(test_full: pd.DataFrame, *, sample_size: int, seed: int) -> pd.DataFrame:
    if sample_size >= len(test_full):
        return test_full.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(test_full), size=int(sample_size), replace=False))
    return test_full.iloc[idx].reset_index(drop=True)


def fit_probe(cal_df: pd.DataFrame, feature_cols: list[str], *, representation: str) -> dict[str, Any]:
    req = feature_cols + ["error"]
    mask = cal_df[req].notna().all(axis=1).to_numpy()
    x_cal = cal_df.loc[mask, feature_cols].to_numpy(dtype=np.float64)
    y_cal = cal_df.loc[mask, "error"].to_numpy(dtype=int)
    if len(x_cal) < 10 or len(np.unique(y_cal)) < 2:
        raise ValueError(f"Invalid calibration data for {representation}")

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=LOGISTIC_C, max_iter=5000, random_state=42)),
        ]
    )
    pipe.fit(x_cal, y_cal)
    cal_scores = pipe.predict_proba(x_cal)[:, 1]
    return {
        "representation": representation,
        "feature_cols": feature_cols,
        "n_cal": int(len(x_cal)),
        "intercept": float(pipe.named_steps["clf"].intercept_[0]),
        "pipeline": pipe,
        "cal_auroc": float(roc_auc_score(y_cal, cal_scores)),
        "cal_auprc": float(average_precision_score(y_cal, cal_scores)),
    }


def eval_probe(fitted: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    feature_cols = fitted["feature_cols"]
    req = feature_cols + ["error"]
    mask = df[req].notna().all(axis=1).to_numpy()
    x = df.loc[mask, feature_cols].to_numpy(dtype=np.float64)
    y = df.loc[mask, "error"].to_numpy(dtype=int)
    scores = fitted["pipeline"].predict_proba(x)[:, 1]
    return {
        **fitted,
        "n_eval": int(len(y)),
        "auroc": float(roc_auc_score(y, scores)),
        "auprc": float(average_precision_score(y, scores)),
        "scores": scores,
        "errors": y,
        "entropy": df.loc[mask, "normalized_entropy"].to_numpy(dtype=float),
    }


def eval_entropy(df: pd.DataFrame) -> dict[str, Any]:
    mask = df[["normalized_entropy", "error"]].notna().all(axis=1).to_numpy()
    y = df.loc[mask, "error"].to_numpy(dtype=int)
    h = df.loc[mask, "normalized_entropy"].to_numpy(dtype=float)
    return {
        "n_eval": int(len(y)),
        "auroc": float(roc_auc_score(y, h)),
        "auprc": float(average_precision_score(y, h)),
        "scores": h,
        "errors": y,
        "entropy": h,
    }


def bootstrap_delta_ci(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        if len(np.unique(y_b)) < 2:
            continue
        deltas.append(roc_auc_score(y_b, score_b[idx]) - roc_auc_score(y_b, score_a[idx]))
    if not deltas:
        return {"delta_mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    arr = np.asarray(deltas)
    return {
        "delta_mean": float(arr.mean()),
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
    }


def bootstrap_seed_ci(per_seed_deltas: list[float]) -> dict[str, float]:
    if not per_seed_deltas:
        return {"delta_mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    arr = np.asarray(per_seed_deltas)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = [float(arr[rng.integers(0, len(arr), size=len(arr))].mean()) for _ in range(N_BOOTSTRAP)]
    return {
        "delta_mean": float(arr.mean()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
    }


def confident_wrong(df: pd.DataFrame, *, score_col: str, h_quantile: float = 0.25) -> dict[str, float]:
    sub = df.dropna(subset=["normalized_entropy", score_col, "error"]).copy()
    h_thr = float(sub["normalized_entropy"].quantile(h_quantile))
    low_h = sub[sub["normalized_entropy"] <= h_thr]
    if low_h.empty:
        return {"h_threshold": h_thr, "n_low_h": 0}
    s_thr = float(low_h[score_col].median())
    high = low_h[low_h[score_col] > s_thr]
    low = low_h[low_h[score_col] <= s_thr]
    return {
        "h_threshold": h_thr,
        "score_threshold": s_thr,
        "n_low_h": int(len(low_h)),
        "p_error_score_high": float(high["error"].mean()) if len(high) else float("nan"),
        "p_error_score_low": float(low["error"].mean()) if len(low) else float("nan"),
        "auroc_score_given_low_h": float(roc_auc_score(low_h["error"], low_h[score_col])),
    }


def selective_risk_table(y: np.ndarray, scores: np.ndarray, *, name: str) -> list[dict[str, float]]:
    rows = []
    for cov in COVERAGES:
        m = deferral_metrics_at_coverage(y, scores, target_coverage=cov)
        rows.append(
            {
                "scorer": name,
                "coverage": cov,
                "deferral_rate": 1.0 - cov,
                "selective_risk": m["R"],
                "error_capture": m["EC"],
            }
        )
    return rows


def load_frames() -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    cal_frames: dict[int, pd.DataFrame] = {}
    test_frames: dict[int, pd.DataFrame] = {}
    for seed in SEEDS:
        cal_path = FEATURE_CACHE / f"seed{seed}" / "cal_features.csv"
        test_path = FEATURE_CACHE / f"seed{seed}" / "test_features_full.csv"
        cal_frames[seed] = pd.read_csv(cal_path)
        test_full = pd.read_csv(test_path)
        test_frames[seed] = subsample_test_df(
            test_full, sample_size=TEST_SAMPLE_SIZE, seed=RANDOM_SEED + seed
        )
    return cal_frames, test_frames


def run_analysis() -> dict[str, Any]:
    cal_frames, test_frames = load_frames()

    phase1_rows: list[dict[str, Any]] = []
    phase1_cal_rows: list[dict[str, Any]] = []
    fitted_by_seed: dict[int, dict[str, dict[str, Any]]] = {}

    for seed in SEEDS:
        cal_df = cal_frames[seed]
        test_df = test_frames[seed]
        fitted_by_seed[seed] = {}
        for rep, cols in [("magnitude", MAG_COLS), ("full_vector", FULL_COLS)]:
            fit = fit_probe(cal_df, cols, representation=rep)
            test_eval = eval_probe(fit, test_df)
            fitted_by_seed[seed][rep] = test_eval
            phase1_cal_rows.append(
                {
                    "seed": seed,
                    "representation": rep,
                    "cal_auroc": fit["cal_auroc"],
                    "cal_auprc": fit["cal_auprc"],
                }
            )
            phase1_rows.append(
                {
                    "seed": seed,
                    "representation": rep,
                    "n_cal": fit["n_cal"],
                    "n_test": test_eval["n_eval"],
                    "cal_auroc": fit["cal_auroc"],
                    "test_auroc": test_eval["auroc"],
                    "test_auprc": test_eval["auprc"],
                }
            )

    cal_df_all = pd.DataFrame(phase1_cal_rows)
    mag_cal_mean = float(cal_df_all.loc[cal_df_all["representation"] == "magnitude", "cal_auroc"].mean())
    full_cal_mean = float(cal_df_all.loc[cal_df_all["representation"] == "full_vector", "cal_auroc"].mean())
    if full_cal_mean >= mag_cal_mean:
        x_best_name = "full_vector"
        x_best_cols = FULL_COLS
    else:
        x_best_name = "magnitude"
        x_best_cols = MAG_COLS

    phase1_df = pd.DataFrame(phase1_rows)
    mag_test_mean = float(phase1_df.loc[phase1_df["representation"] == "magnitude", "test_auroc"].mean())
    full_test_mean = float(phase1_df.loc[phase1_df["representation"] == "full_vector", "test_auroc"].mean())

    phase3_rows: list[dict[str, Any]] = []
    level_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    selective_rows: list[dict[str, Any]] = []
    confident_rows: list[dict[str, Any]] = []

    pooled_y: list[np.ndarray] = []
    pooled_h: list[np.ndarray] = []
    pooled_z: list[np.ndarray] = []
    per_seed_delta_zh: list[float] = []
    per_seed_delta_full_mag: list[float] = []

    z_combined_by_seed: dict[int, pd.DataFrame] = {}

    for seed in SEEDS:
        cal_df = cal_frames[seed]
        test_df = test_frames[seed].copy()

        z_fit = fit_probe(cal_df, x_best_cols, representation=f"z_combined_{x_best_name}")
        z_eval = eval_probe(z_fit, test_df)
        h_eval = eval_entropy(test_df)

        test_df = test_df.copy()
        mask = test_df[x_best_cols + ["error"]].notna().all(axis=1).to_numpy()
        test_df.loc[mask, "z_combined_score"] = z_eval["scores"]
        z_combined_by_seed[seed] = test_df

        phase3_rows.extend(
            [
                {"seed": seed, "model": "entropy_H", "auroc": h_eval["auroc"], "auprc": h_eval["auprc"]},
                {"seed": seed, "model": "z_combined", "auroc": z_eval["auroc"], "auprc": z_eval["auprc"]},
            ]
        )
        per_seed_delta_zh.append(z_eval["auroc"] - h_eval["auroc"])
        per_seed_delta_full_mag.append(
            fitted_by_seed[seed]["full_vector"]["auroc"] - fitted_by_seed[seed]["magnitude"]["auroc"]
        )

        pooled_y.append(z_eval["errors"])
        pooled_h.append(h_eval["scores"])
        pooled_z.append(z_eval["scores"])

        for j in range(1, 5):
            cols = level_cols(j)
            lev = eval_probe(fit_probe(cal_df, cols, representation=f"level_{j}"), test_df)
            level_rows.append(
                {"seed": seed, "level": j, "auroc": lev["auroc"], "auprc": lev["auprc"], "n_features": len(cols)}
            )

        for max_l in range(1, 5):
            cols = cumulative_cols(max_l)
            ab = eval_probe(fit_probe(cal_df, cols, representation=f"cumulative_{max_l}"), test_df)
            ablation_rows.append(
                {
                    "seed": seed,
                    "levels": f"z1..z{max_l}",
                    "max_level": max_l,
                    "auroc": ab["auroc"],
                    "auprc": ab["auprc"],
                }
            )

        for name, scores in [("H", h_eval["scores"]), ("z_combined", z_eval["scores"])]:
            selective_rows.extend(selective_risk_table(z_eval["errors"], scores, name=name))

        cw_h = confident_wrong(test_df, score_col="normalized_entropy")
        cw_z = confident_wrong(test_df, score_col="z_combined_score")
        confident_rows.append({"seed": seed, "scorer": "H", **{k: v for k, v in cw_h.items() if k != "auroc_score_given_low_h"}})
        confident_rows.append(
            {
                "seed": seed,
                "scorer": "z_combined",
                **{k: v for k, v in cw_z.items() if k != "auroc_score_given_low_h"},
                "auroc_given_low_h": cw_z.get("auroc_score_given_low_h", float("nan")),
            }
        )

    y_pool = np.concatenate(pooled_y)
    h_pool = np.concatenate(pooled_h)
    z_pool = np.concatenate(pooled_z)

    bootstrap_z_vs_h = bootstrap_delta_ci(y_pool, h_pool, z_pool)
    bootstrap_full_vs_mag = bootstrap_delta_ci(
        y_pool,
        np.concatenate([fitted_by_seed[s]["magnitude"]["scores"] for s in SEEDS]),
        np.concatenate([fitted_by_seed[s]["full_vector"]["scores"] for s in SEEDS]),
    )
    bootstrap_z_vs_h_seed = bootstrap_seed_ci(per_seed_delta_zh)
    bootstrap_full_vs_mag_seed = bootstrap_seed_ci(per_seed_delta_full_mag)

    z_auroc_mean = float(np.mean([r["auroc"] for r in phase3_rows if r["model"] == "z_combined"]))
    h_auroc_mean = float(np.mean([r["auroc"] for r in phase3_rows if r["model"] == "entropy_H"]))

    results = {
        "config": {
            "seeds": SEEDS,
            "test_sample_size": TEST_SAMPLE_SIZE,
            "random_seed": RANDOM_SEED,
            "logistic_C": LOGISTIC_C,
            "x_best_selection": "calibration_auroc_mean_across_seeds",
        },
        "phase1_decision": {
            "x_best": x_best_name,
            "mean_cal_auroc_magnitude": mag_cal_mean,
            "mean_cal_auroc_full_vector": full_cal_mean,
            "mean_test_auroc_magnitude": mag_test_mean,
            "mean_test_auroc_full_vector": full_test_mean,
            "full_beats_magnitude_on_cal_auroc": full_cal_mean >= mag_cal_mean,
        },
        "phase1_per_seed": phase1_df.to_dict(orient="records"),
        "phase3_summary": {
            "x_best": x_best_name,
            "mean_auroc_z_combined": z_auroc_mean,
            "mean_auroc_H": h_auroc_mean,
            "mean_delta_auroc": z_auroc_mean - h_auroc_mean,
            "bootstrap_delta_auroc_pooled": bootstrap_z_vs_h,
            "bootstrap_delta_auroc_per_seed": bootstrap_z_vs_h_seed,
        },
        "phase1_bootstrap_full_vs_mag_pooled": bootstrap_full_vs_mag,
        "phase1_bootstrap_full_vs_mag_per_seed": bootstrap_full_vs_mag_seed,
        "phase3_per_seed": phase3_rows,
        "level_per_seed": level_rows,
        "ablation_per_seed": ablation_rows,
        "selective_risk": selective_rows,
        "confident_wrong": confident_rows,
    }

    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROBE_DIR / "probe_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    phase1_df.to_csv(PROBE_DIR / "phase1_per_seed.csv", index=False)
    pd.DataFrame(phase3_rows).to_csv(PROBE_DIR / "phase3_per_seed.csv", index=False)
    pd.DataFrame(level_rows).to_csv(PROBE_DIR / "level_per_seed.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(PROBE_DIR / "ablation_per_seed.csv", index=False)
    pd.DataFrame(selective_rows).to_csv(PROBE_DIR / "selective_risk.csv", index=False)
    pd.DataFrame(confident_rows).to_csv(PROBE_DIR / "confident_wrong.csv", index=False)

    generate_figures(results, y_pool, h_pool, z_pool, z_combined_by_seed[42])
    return results


def generate_figures(
    results: dict[str, Any],
    y: np.ndarray,
    h: np.ndarray,
    z: np.ndarray,
    scatter_df: pd.DataFrame,
) -> None:
    PUB_IMAGES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})

    # ROC + PR (pooled)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for scores, label, color in [(h, r"$\tilde H$", "#DD8452"), (z, r"$z_{\mathrm{combined}}$", "#4C72B0")]:
        fpr, tpr, _ = roc_curve(y, scores)
        prec, rec, _ = precision_recall_curve(y, scores)
        axes[0].plot(fpr, tpr, label=f"{label} (AUROC={roc_auc_score(y, scores):.3f})", color=color, lw=2)
        axes[1].plot(rec, prec, label=f"{label} (AUPRC={average_precision_score(y, scores):.3f})", color=color, lw=2)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC curves (pooled test)")
    axes[0].legend(loc="lower right", fontsize=7)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("PR curves (pooled test)")
    axes[1].legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(PUB_IMAGES / "roc_pr_curves.pdf")
    plt.close(fig)

    # Risk-coverage
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for scores, label, color in [(h, r"$\tilde H$", "#DD8452"), (z, r"$z_{\mathrm{combined}}$", "#4C72B0")]:
        curve = risk_coverage_curve(y, scores)
        ax.plot(curve["coverage"], curve["risk"], "-o", ms=3, label=label, color=color)
    ax.set_xlabel("Coverage (fraction accepted)")
    ax.set_ylabel("Selective risk (error rate among accepted)")
    ax.set_title("Risk--coverage curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PUB_IMAGES / "risk_coverage_curves.pdf")
    plt.close(fig)

    # Phase 1 bar chart (test AUROC, three seeds)
    phase1 = pd.DataFrame(results["phase1_per_seed"])
    seeds = SEEDS
    x = np.arange(len(seeds))
    w = 0.35
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    mag = [phase1[(phase1.seed == s) & (phase1.representation == "magnitude")]["test_auroc"].iloc[0] for s in seeds]
    full = [phase1[(phase1.seed == s) & (phase1.representation == "full_vector")]["test_auroc"].iloc[0] for s in seeds]
    ax.bar(x - w / 2, mag, w, label=r"$X_{\mathrm{mag}}$", color="#DD8452")
    ax.bar(x + w / 2, full, w, label=r"$X_{\mathrm{full}}$", color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_ylabel("Test AUROC")
    ax.set_ylim(0.6, 0.78)
    ax.legend(loc="lower right")
    ax.set_title("Full vectors vs magnitudes (test)")
    fig.tight_layout()
    fig.savefig(PUB_IMAGES / "probe_phase1_auroc.pdf")
    plt.close(fig)

    # Phase 3 bar chart
    phase3 = pd.DataFrame(results["phase3_per_seed"])
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    h_vals = [phase3[(phase3.seed == s) & (phase3.model == "entropy_H")]["auroc"].iloc[0] for s in seeds]
    z_vals = [phase3[(phase3.seed == s) & (phase3.model == "z_combined")]["auroc"].iloc[0] for s in seeds]
    ax.bar(x - w / 2, h_vals, w, label=r"$\tilde H$", color="#DD8452")
    ax.bar(x + w / 2, z_vals, w, label=r"$z_{\mathrm{combined}}$", color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_ylabel("Test AUROC")
    ax.set_ylim(0.55, 0.78)
    ax.legend(loc="lower right")
    ax.set_title(r"$z_{\mathrm{combined}}$ vs $\tilde H$ (test)")
    fig.tight_layout()
    fig.savefig(PUB_IMAGES / "probe_phase3_vs_entropy.pdf")
    plt.close(fig)

    # Level ablation (mean over seeds)
    ab = pd.DataFrame(results["ablation_per_seed"])
    ab_mean = ab.groupby("max_level")[["auroc", "auprc"]].mean()
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.plot(ab_mean.index, ab_mean["auroc"], "-o", color="#4C72B0", label="AUROC")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("Cumulative levels")
    ax.set_ylabel("Test AUROC (mean over seeds)")
    ax.set_title(r"Cumulative ablation $z_1 \rightarrow \cdots \rightarrow z_4$")
    ax.set_ylim(0.6, 0.78)
    fig.tight_layout()
    fig.savefig(PUB_IMAGES / "level_ablation.pdf")
    plt.close(fig)

    # Individual levels
    lev = pd.DataFrame(results["level_per_seed"])
    lev_mean = lev.groupby("level")["auroc"].mean()
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.bar([str(i) for i in lev_mean.index], lev_mean.values, color="#55A868")
    ax.set_xlabel("Memory level $j$")
    ax.set_ylabel("Test AUROC (mean over seeds)")
    ax.set_title("Per-level logistic probes")
    ax.set_ylim(0.5, 0.75)
    fig.tight_layout()
    fig.savefig(PUB_IMAGES / "level_individual.pdf")
    plt.close(fig)

    # H vs z_combined scatter
    sub = scatter_df.dropna(subset=["normalized_entropy", "z_combined_score", "error"])
    rng = np.random.default_rng(42)
    n_show = min(800, len(sub))
    idx = rng.choice(len(sub), size=n_show, replace=False)
    sub = sub.iloc[idx]
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    colors = np.where(sub["error"].astype(bool), "#C44E52", "#4C72B0")
    ax.scatter(sub["normalized_entropy"], sub["z_combined_score"], c=colors, s=8, alpha=0.5, linewidths=0)
    ax.set_xlabel(r"Normalized entropy $\tilde H$")
    ax.set_ylabel(r"$z_{\mathrm{combined}}$ failure score")
    ax.set_title("ID test (seed 42; subset)")
    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C72B0", markersize=6, label="Correct"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#C44E52", markersize=6, label="Error"),
        ],
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(PUB_IMAGES / "entropy_vs_z_combined.pdf")
    plt.close(fig)

    # Confident-wrong bar
    cw = pd.DataFrame(results["confident_wrong"])
    cw_z = cw[cw["scorer"] == "z_combined"]
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    x = np.arange(len(SEEDS))
    w = 0.35
    ax.bar(x - w / 2, cw_z["p_error_score_low"], w, label="Low $z$ (confident)", color="#4C72B0")
    ax.bar(x + w / 2, cw_z["p_error_score_high"], w, label="High $z$ (confident)", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_ylabel(r"$P(\mathrm{error}\mid \tilde H$ low$)$")
    ax.set_title("Confident-wrong region: error rate by memory score")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PUB_IMAGES / "confident_wrong.pdf")
    plt.close(fig)

    # Reconstruction from verdict
    verdict_path = REPO_ROOT / "research" / "associative_memory_fft" / "artifacts" / "verdict_draft.json"
    if verdict_path.exists():
        data = json.loads(verdict_path.read_text())
        levels = [1, 2, 3, 4]
        cosine = [data["reconstruction"]["cosine"][str(j)] for j in levels]
        mse = [data["reconstruction"]["mse"][str(j)] for j in levels]
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))
        axes[0].bar([str(j) for j in levels], cosine, color="#4C72B0")
        axes[0].set_xlabel("Level $j$")
        axes[0].set_ylabel("Cosine")
        axes[1].bar([str(j) for j in levels], mse, color="#DD8452")
        axes[1].set_xlabel("Level $j$")
        axes[1].set_ylabel("MSE")
        fig.tight_layout()
        fig.savefig(PUB_IMAGES / "reconstruction_by_level.pdf")
        plt.close(fig)

    print(f"Wrote figures to {PUB_IMAGES}")


if __name__ == "__main__":
    out = run_analysis()
    s = out["phase3_summary"]
    print(f"z_combined AUROC={s['mean_auroc_z_combined']:.4f}  H={s['mean_auroc_H']:.4f}  delta={s['mean_delta_auroc']:.4f}")
    print(f"Bootstrap delta (pooled): {s['bootstrap_delta_auroc_pooled']}")
