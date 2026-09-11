"""Generate correctness-memory publication figures (plots 3--7) with *_correctness suffix.

Data source: feature caches from notebooks/dermamnist_correctness_memory_full.ipynb
(Part II) and probe protocol from notebooks/memory_vector_vs_magnitude_correctness_probe.ipynb.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from research.associative_memory_fft.probe_experiments.probe_analysis import (
    RANDOM_SEED,
    SEEDS,
    TEST_SAMPLE_SIZE,
    confident_wrong,
    deferral_metrics_at_coverage,
    eval_entropy,
    eval_probe,
    fit_probe,
    risk_coverage_curve,
    subsample_test_df,
)
from research.correctness_memory_fft.probe_experiments.correctness_probe_analysis import (
    FEATURE_CACHE,
    select_z_combined_cols,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PUB_IMAGES = Path(__file__).resolve().parent / "images"

# Table reference values in main.tex (seed-mean unless noted).
TABLE_PROBE_PHASE3 = {"H": 0.664, "z": 0.753, "H_auprc": 0.497, "z_auprc": 0.606}
TABLE_SELECTIVE_20 = {"H": 0.280, "z": 0.256}
TABLE_CONFIDENT_MEAN = {"low_z": 0.095, "high_z": 0.272}
TABLE_CONFIDENT_SEED42 = {"low_z": 0.10, "high_z": 0.32, "auroc_low_h": 0.715}


def load_probe_frames() -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame], list[str]]:
    """Match memory_vector_vs_magnitude_correctness_probe subsampling (fixed RANDOM_SEED)."""
    cal_frames: dict[int, pd.DataFrame] = {}
    test_frames: dict[int, pd.DataFrame] = {}
    for seed in SEEDS:
        seed_dir = FEATURE_CACHE / f"seed{seed}"
        cal_path = seed_dir / "cal_features.csv"
        test_path = seed_dir / "test_features_full.csv"
        if not cal_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                "Missing correctness feature cache. Run "
                "notebooks/dermamnist_correctness_memory_full.ipynb Part II first."
            )
        cal_frames[seed] = pd.read_csv(cal_path)
        test_full = pd.read_csv(test_path)
        test_frames[seed] = subsample_test_df(
            test_full, sample_size=TEST_SAMPLE_SIZE, seed=RANDOM_SEED
        )
    z_cols = select_z_combined_cols(cal_frames)
    return cal_frames, test_frames, z_cols


def run_correctness_probe(
    cal_frames: dict[int, pd.DataFrame],
    test_frames: dict[int, pd.DataFrame],
    z_cols: list[str],
) -> dict[str, Any]:
    phase3_rows: list[dict[str, Any]] = []
    selective_rows: list[dict[str, Any]] = []
    confident_rows: list[dict[str, Any]] = []
    pooled_y: list[np.ndarray] = []
    pooled_h: list[np.ndarray] = []
    pooled_z: list[np.ndarray] = []
    z_combined_by_seed: dict[int, pd.DataFrame] = {}

    for seed in SEEDS:
        cal_df = cal_frames[seed]
        test_df = test_frames[seed].copy()
        z_fit = fit_probe(cal_df, z_cols, representation="z_combined")
        z_eval = eval_probe(z_fit, test_df)
        h_eval = eval_entropy(test_df)

        mask = test_df[z_cols + ["error"]].notna().all(axis=1).to_numpy()
        test_df = test_df.copy()
        test_df.loc[mask, "z_combined_score"] = z_eval["scores"]
        z_combined_by_seed[seed] = test_df

        phase3_rows.extend(
            [
                {"seed": seed, "model": "entropy_H", "auroc": h_eval["auroc"], "auprc": h_eval["auprc"]},
                {"seed": seed, "model": "z_combined", "auroc": z_eval["auroc"], "auprc": z_eval["auprc"]},
            ]
        )
        pooled_y.append(z_eval["errors"])
        pooled_h.append(h_eval["scores"])
        pooled_z.append(z_eval["scores"])

        for defer in (0.20, 0.10, 0.05):
            cov = 1.0 - defer
            m_h = deferral_metrics_at_coverage(z_eval["errors"], h_eval["scores"], target_coverage=cov)
            m_z = deferral_metrics_at_coverage(z_eval["errors"], z_eval["scores"], target_coverage=cov)
            selective_rows.append(
                {"seed": seed, "scorer": "H", "deferral": defer, "selective_risk": m_h["R"]}
            )
            selective_rows.append(
                {"seed": seed, "scorer": "z_combined", "deferral": defer, "selective_risk": m_z["R"]}
            )

        cw_z = confident_wrong(test_df, score_col="z_combined_score")
        confident_rows.append(
            {
                "seed": seed,
                "scorer": "z_combined",
                "p_error_score_low": cw_z["p_error_score_low"],
                "p_error_score_high": cw_z["p_error_score_high"],
                "auroc_given_low_h": cw_z.get("auroc_score_given_low_h", float("nan")),
            }
        )

    y_pool = np.concatenate(pooled_y)
    h_pool = np.concatenate(pooled_h)
    z_pool = np.concatenate(pooled_z)
    phase3_df = pd.DataFrame(phase3_rows)

    return {
        "phase3_per_seed": phase3_df,
        "selective_risk": pd.DataFrame(selective_rows),
        "confident_wrong": pd.DataFrame(confident_rows),
        "pooled": {"y": y_pool, "h": h_pool, "z": z_pool},
        "scatter_df": z_combined_by_seed[42],
        "pooled_metrics": {
            "H_auroc": float(roc_auc_score(y_pool, h_pool)),
            "H_auprc": float(average_precision_score(y_pool, h_pool)),
            "z_auroc": float(roc_auc_score(y_pool, z_pool)),
            "z_auprc": float(average_precision_score(y_pool, z_pool)),
        },
        "seed_mean": {
            "H_auroc": float(phase3_df.loc[phase3_df["model"] == "entropy_H", "auroc"].mean()),
            "z_auroc": float(phase3_df.loc[phase3_df["model"] == "z_combined", "auroc"].mean()),
            "H_auprc": float(phase3_df.loc[phase3_df["model"] == "entropy_H", "auprc"].mean()),
            "z_auprc": float(phase3_df.loc[phase3_df["model"] == "z_combined", "auprc"].mean()),
        },
    }


def generate_correctness_figures(results: dict[str, Any]) -> dict[str, float]:
    PUB_IMAGES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})

    y = results["pooled"]["y"]
    h = results["pooled"]["h"]
    z = results["pooled"]["z"]
    pm = results["pooled_metrics"]
    phase3 = results["phase3_per_seed"]
    seeds = SEEDS
    x = np.arange(len(seeds))
    w = 0.35

    # Fig 3: entropy vs z_combined scatter
    sub = results["scatter_df"].dropna(subset=["normalized_entropy", "z_combined_score", "error"])
    rng = np.random.default_rng(42)
    n_show = min(800, len(sub))
    idx = rng.choice(len(sub), size=n_show, replace=False)
    sub = sub.iloc[idx]
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    colors = np.where(sub["error"].astype(bool), "#C44E52", "#4C72B0")
    ax.scatter(sub["normalized_entropy"], sub["z_combined_score"], c=colors, s=8, alpha=0.5, linewidths=0)
    ax.set_xlabel(r"Normalized entropy $\tilde H$")
    ax.set_ylabel(r"$z_{\mathrm{combined}}$ failure score")
    ax.set_title("ID test (seed 42; subset; correctness memory)")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C72B0", markersize=6, label="Correct"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#C44E52", markersize=6, label="Error"),
        ],
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(PUB_IMAGES / "entropy_vs_z_combined_correctness.pdf")
    plt.close(fig)

    # Fig 4: phase3 per-seed AUROC bars (+ seed-mean reference lines)
    sm = results["seed_mean"]
    h_vals = [phase3[(phase3.seed == s) & (phase3.model == "entropy_H")]["auroc"].iloc[0] for s in seeds]
    z_vals = [phase3[(phase3.seed == s) & (phase3.model == "z_combined")]["auroc"].iloc[0] for s in seeds]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.bar(x - w / 2, h_vals, w, label=r"$\tilde H$", color="#DD8452")
    ax.bar(x + w / 2, z_vals, w, label=r"$z_{\mathrm{combined}}$", color="#4C72B0")
    ax.axhline(sm["H_auroc"], color="#DD8452", ls="--", lw=1.0, alpha=0.8)
    ax.axhline(sm["z_auroc"], color="#4C72B0", ls="--", lw=1.0, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_ylabel("Test AUROC")
    ax.set_ylim(0.55, 0.80)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(
        r"$z_{\mathrm{combined}}$ vs $\tilde H$ (per-seed; dashed = table mean)"
    )
    fig.tight_layout()
    fig.savefig(PUB_IMAGES / "probe_phase3_vs_entropy_correctness.pdf")
    plt.close(fig)

    # Fig 5: pooled ROC/PR
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for scores, label, color, auroc, auprc in [
        (h, r"$\tilde H$", "#DD8452", pm["H_auroc"], pm["H_auprc"]),
        (z, r"$z_{\mathrm{combined}}$", "#4C72B0", pm["z_auroc"], pm["z_auprc"]),
    ]:
        fpr, tpr, _ = roc_curve(y, scores)
        prec, rec, _ = precision_recall_curve(y, scores)
        axes[0].plot(fpr, tpr, label=f"{label} (AUROC={auroc:.3f})", color=color, lw=2)
        axes[1].plot(rec, prec, label=f"{label} (AUPRC={auprc:.3f})", color=color, lw=2)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC curves (pooled test, $n{=}6000$)")
    axes[0].legend(loc="lower right", fontsize=7)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("PR curves (pooled test, $n{=}6000$)")
    axes[1].legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(PUB_IMAGES / "roc_pr_curves_correctness.pdf")
    plt.close(fig)

    # Fig 6: risk-coverage (pooled)
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for scores, label, color in [(h, r"$\tilde H$", "#DD8452"), (z, r"$z_{\mathrm{combined}}$", "#4C72B0")]:
        curve = risk_coverage_curve(y, scores)
        ax.plot(curve["coverage"], curve["risk"], "-o", ms=3, label=label, color=color)
    ax.set_xlabel("Coverage (fraction accepted)")
    ax.set_ylabel("Selective risk (error rate among accepted)")
    ax.set_title("Risk--coverage (pooled test; selective-risk table = seed means)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PUB_IMAGES / "risk_coverage_curves_correctness.pdf")
    plt.close(fig)

    # Fig 7: confident-wrong bars per seed
    cw = results["confident_wrong"]
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.bar(x - w / 2, cw["p_error_score_low"], w, label="Low $z$", color="#4C72B0")
    ax.bar(x + w / 2, cw["p_error_score_high"], w, label="High $z$", color="#C44E52")
    ax.axhline(TABLE_CONFIDENT_MEAN["low_z"], color="#4C72B0", ls="--", lw=1.0, alpha=0.8)
    ax.axhline(TABLE_CONFIDENT_MEAN["high_z"], color="#C44E52", ls="--", lw=1.0, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_ylabel(r"$P(\mathrm{error}\mid \tilde H$ low$)$")
    ax.set_title("Confident-wrong region (per-seed; dashed = table mean)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PUB_IMAGES / "confident_wrong_correctness.pdf")
    plt.close(fig)

    print(f"Wrote *_correctness figures to {PUB_IMAGES}")
    return pm


def validate_against_tables(results: dict[str, Any]) -> None:
    sm = results["seed_mean"]
    pm = results["pooled_metrics"]
    sel = results["selective_risk"]
    cw = results["confident_wrong"]
    sel20 = sel[sel["deferral"] == 0.20].groupby("scorer")["selective_risk"].mean()

    print("\n=== Cross-validation vs main.tex tables ===")
    print(
        f"Table probe-phase3 seed-mean AUROC: H={TABLE_PROBE_PHASE3['H']:.3f}, "
        f"z={TABLE_PROBE_PHASE3['z']:.3f}"
    )
    print(f"  Figure 4 / computed seed-mean:      H={sm['H_auroc']:.3f}, z={sm['z_auroc']:.3f}")
    print(
        f"Table probe-phase3 seed-mean AUPRC: H={TABLE_PROBE_PHASE3['H_auprc']:.3f}, "
        f"z={TABLE_PROBE_PHASE3['z_auprc']:.3f}"
    )
    print(f"  Computed seed-mean AUPRC:           H={sm['H_auprc']:.3f}, z={sm['z_auprc']:.3f}")
    print(
        f"Pooled ROC legend (Fig 5): H AUROC={pm['H_auroc']:.3f}, z AUROC={pm['z_auroc']:.3f} "
        f"(differs from seed-mean for H; see Sec.~exp-metrics)"
    )
    print(
        f"Table selective 20% def.: H={TABLE_SELECTIVE_20['H']:.3f}, z={TABLE_SELECTIVE_20['z']:.3f}"
    )
    print(
        f"  Computed seed-mean selective:       H={sel20['H']:.3f}, z={sel20['z_combined']:.3f}"
    )
    print("  Fig 6 risk--coverage uses pooled curves (not tabulated point values).")
    print(
        f"Table confident mean: low={TABLE_CONFIDENT_MEAN['low_z']:.3f}, "
        f"high={TABLE_CONFIDENT_MEAN['high_z']:.3f}"
    )
    print(
        f"  Computed confident mean:            low={cw['p_error_score_low'].mean():.3f}, "
        f"high={cw['p_error_score_high'].mean():.3f}"
    )
    s42 = cw[cw["seed"] == 42].iloc[0]
    print(
        f"Table confident seed 42: low={TABLE_CONFIDENT_SEED42['low_z']:.2f}, "
        f"high={TABLE_CONFIDENT_SEED42['high_z']:.2f}, AUROC|low H={TABLE_CONFIDENT_SEED42['auroc_low_h']:.3f}"
    )
    print(
        f"  Computed seed 42:                   low={s42['p_error_score_low']:.2f}, "
        f"high={s42['p_error_score_high']:.2f}, AUROC|low H={s42['auroc_given_low_h']:.3f}"
    )


def main() -> None:
    cal_frames, test_frames, z_cols = load_probe_frames()
    results = run_correctness_probe(cal_frames, test_frames, z_cols)
    generate_correctness_figures(results)
    validate_against_tables(results)
    out_path = PUB_IMAGES / "correctness_figure_metrics.json"
    payload = {
        "seed_mean": results["seed_mean"],
        "pooled_metrics": results["pooled_metrics"],
        "phase3_per_seed": results["phase3_per_seed"].to_dict(orient="records"),
        "selective_risk": results["selective_risk"].to_dict(orient="records"),
        "confident_wrong": results["confident_wrong"].to_dict(orient="records"),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote metrics snapshot to {out_path}")


if __name__ == "__main__":
    main()
