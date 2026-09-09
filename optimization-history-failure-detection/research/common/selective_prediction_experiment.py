"""Final NRO selective-prediction / human-deferral experiment on DermaMNIST."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.common.clinical_failure import fit_combined_scorer, risk_coverage_curve
from research.common.error_prediction import predict_error_probability

EvaluationDomain = Literal["external", "id", "corruption"]
ScorerName = Literal["H", "H+N"]

PRIMARY_DEFERRAL_RATE = 0.20
PRIMARY_COVERAGE = 0.80
SECONDARY_DEFERRAL_RATES = (0.10, 0.20, 0.30)
CURVE_COVERAGES = tuple(np.round(np.arange(0.50, 1.01, 0.05), 2))


@dataclass
class SelectivePredictionConfig:
    repo_root: Path | None = None
    source_experiment_dir: Path | None = None
    output_dir: Path | None = None
    seeds: tuple[int, ...] = (42, 123, 456)
    task: str = "dermamnist"
    evaluation_domain: EvaluationDomain = "external"
    h_col: str = "normalized_entropy"
    n_col: str = "memory_novelty"
    num_classes: int = 7
    primary_coverage: float = PRIMARY_COVERAGE
    secondary_deferral_rates: tuple[float, ...] = SECONDARY_DEFERRAL_RATES
    n_bootstrap: int = 2000
    bootstrap_seed: int = 20260830
    ci_level: float = 0.95
    show_plots: bool = True
    save_artifacts: bool = True

    def resolve_paths(self, cwd: Path | None = None) -> SelectivePredictionConfig:
        root = self.repo_root
        if root is None:
            root = Path(cwd or Path.cwd()).resolve()
            if not (root / "research").exists() and (root.parent / "research").exists():
                root = root.parent
        self.repo_root = root
        if self.source_experiment_dir is None:
            self.source_experiment_dir = root / "research" / "final_experiment"
        if self.output_dir is None:
            self.output_dir = root / "research" / "dermamnist_selective_prediction"
        return self


def _ensure_h_column(df: pd.DataFrame, cfg: SelectivePredictionConfig) -> pd.DataFrame:
    out = df.copy()
    if cfg.h_col not in out.columns:
        if "predictive_entropy" not in out.columns:
            raise ValueError(f"Cannot derive {cfg.h_col}")
        out[cfg.h_col] = out["predictive_entropy"].astype(float) / np.log(cfg.num_classes)
    return out


def load_calibration_frames(cfg: SelectivePredictionConfig) -> pd.DataFrame:
    source = cfg.source_experiment_dir
    assert source is not None
    parts = []
    for seed in cfg.seeds:
        path = source / "calibration" / f"{cfg.task}_seed{seed}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        parts.append(pd.read_csv(path))
    return pd.concat(parts, ignore_index=True)


def load_test_frames(cfg: SelectivePredictionConfig) -> pd.DataFrame:
    source = cfg.source_experiment_dir
    assert source is not None
    parts = []
    for seed in cfg.seeds:
        path = source / "scored" / f"{cfg.task}_seed{seed}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        parts.append(df[df["domain"] == cfg.evaluation_domain].copy())
    out = pd.concat(parts, ignore_index=True)
    if out.empty:
        raise ValueError(f"No test rows for domain={cfg.evaluation_domain!r}")
    return out


def prepare_frame(df: pd.DataFrame, cfg: SelectivePredictionConfig) -> pd.DataFrame:
    df = _ensure_h_column(df, cfg)
    out = df.copy()
    out["error"] = (1 - pd.to_numeric(out["correct"], errors="coerce").fillna(0)).astype(int)
    out["H"] = out[cfg.h_col].astype(float)
    out["N"] = out[cfg.n_col].astype(float)
    return out


def attach_failure_scores(
    cal_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    cfg: SelectivePredictionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    cal = prepare_frame(cal_df, cfg)
    ev = prepare_frame(eval_df, cfg)
    combined = fit_combined_scorer(cal, h_col=cfg.h_col, n_col=cfg.n_col)
    cal["S_H"] = cal["H"]
    ev["S_H"] = ev["H"]
    cal["S_HN"] = predict_error_probability(combined, cal, [cfg.h_col, cfg.n_col])
    ev["S_HN"] = predict_error_probability(combined, ev, [cfg.h_col, cfg.n_col])
    return cal, ev, combined


def assert_no_leakage(cal_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    overlap = set(cal_df["sample_id"].astype(str)) & set(test_df["sample_id"].astype(str))
    if overlap:
        raise AssertionError(f"Calibration/test leakage: {len(overlap)} overlapping sample_ids")


def calibrate_deferral_threshold(scores: np.ndarray, *, target_coverage: float) -> float:
    """Return threshold tau: defer samples with score > tau to achieve ~target_coverage retained."""
    valid = np.asarray(scores, dtype=float)
    valid = valid[np.isfinite(valid)]
    if len(valid) == 0:
        return float("nan")
    n = len(valid)
    k = max(1, min(n, int(round(target_coverage * n))))
    sorted_scores = np.sort(valid)
    return float(sorted_scores[k - 1])


def apply_frozen_deferral(
    errors: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    errors = np.asarray(errors, dtype=int)
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(scores)
    errors = errors[mask]
    scores = scores[mask]
    n = len(errors)
    if n == 0:
        return _empty_metrics()

    deferred = scores > threshold
    accepted = ~deferred
    n_accepted = int(accepted.sum())
    n_deferred = int(deferred.sum())
    n_errors = int(errors.sum())
    n_errors_deferred = int(errors[deferred].sum()) if n_deferred else 0
    n_errors_accepted = int(errors[accepted].sum()) if n_accepted else 0

    r_all = float(errors.mean())
    r_selective = float(errors[accepted].mean()) if n_accepted else float("nan")
    acc_selective = float(1.0 - r_selective) if n_accepted else float("nan")
    err_rate_deferred = float(errors[deferred].mean()) if n_deferred else float("nan")

    return {
        "n": n,
        "coverage": float(n_accepted / n),
        "deferral_rate": float(n_deferred / n),
        "selective_risk": r_selective,
        "selective_accuracy": acc_selective,
        "error_rate_deferred": err_rate_deferred,
        "n_deferred": n_deferred,
        "n_accepted": n_accepted,
        "n_errors": n_errors,
        "n_errors_deferred": n_errors_deferred,
        "n_errors_accepted": n_errors_accepted,
        "error_capture": float(n_errors_deferred / n_errors) if n_errors else float("nan"),
        "deferral_precision": float(n_errors_deferred / n_deferred) if n_deferred else float("nan"),
        "risk_all": r_all,
        "risk_reduction": float(r_all - r_selective) if n_accepted else float("nan"),
        "threshold": float(threshold),
    }


def _empty_metrics() -> dict[str, Any]:
    keys = [
        "n", "coverage", "deferral_rate", "selective_risk", "selective_accuracy",
        "error_rate_deferred", "n_deferred", "n_accepted", "n_errors", "n_errors_deferred",
        "n_errors_accepted", "error_capture", "deferral_precision", "risk_all", "risk_reduction", "threshold",
    ]
    return {k: float("nan") for k in keys}


def evaluate_scorer_at_coverage(
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    score_col: str,
    target_coverage: float,
    scorer: ScorerName,
) -> dict[str, Any]:
    threshold = calibrate_deferral_threshold(cal_df[score_col].to_numpy(), target_coverage=target_coverage)
    metrics = apply_frozen_deferral(test_df["error"].to_numpy(), test_df[score_col].to_numpy(), threshold)
    metrics["scorer"] = scorer
    metrics["target_coverage"] = target_coverage
    metrics["target_deferral_rate"] = 1.0 - target_coverage
    return metrics


def build_risk_coverage_from_calibration(
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    score_col: str,
    scorer: ScorerName,
    coverages: tuple[float, ...] = CURVE_COVERAGES,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cov in coverages:
        m = evaluate_scorer_at_coverage(cal_df, test_df, score_col=score_col, target_coverage=cov, scorer=scorer)
        rows.append(m)
    return pd.DataFrame(rows)


def bootstrap_paired_deltas(
    test_df: pd.DataFrame,
    *,
    threshold_h: float,
    threshold_hn: float,
    cfg: SelectivePredictionConfig,
) -> pd.DataFrame:
    errors = test_df["error"].to_numpy(dtype=int)
    s_h = test_df["S_H"].to_numpy(dtype=float)
    s_hn = test_df["S_HN"].to_numpy(dtype=float)
    n = len(test_df)
    rng = np.random.default_rng(cfg.bootstrap_seed)

    rows: list[dict[str, float]] = []
    for b in range(cfg.n_bootstrap):
        idx = rng.integers(0, n, size=n)
        m_h = apply_frozen_deferral(errors[idx], s_h[idx], threshold_h)
        m_hn = apply_frozen_deferral(errors[idx], s_hn[idx], threshold_hn)
        rows.append(
            {
                "bootstrap_id": b,
                "delta_risk": m_h["selective_risk"] - m_hn["selective_risk"],
                "delta_error_capture": m_hn["error_capture"] - m_h["error_capture"],
                "delta_deferral_precision": m_hn["deferral_precision"] - m_h["deferral_precision"],
                "risk_H": m_h["selective_risk"],
                "risk_HN": m_hn["selective_risk"],
                "error_capture_H": m_h["error_capture"],
                "error_capture_HN": m_hn["error_capture"],
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


def preregistered_verdict(
    primary_h: dict[str, Any],
    primary_hn: dict[str, Any],
    boot_df: pd.DataFrame,
    *,
    ci_level: float,
) -> dict[str, Any]:
    delta_risk_mean, delta_risk_lo, delta_risk_hi = bootstrap_ci(boot_df["delta_risk"].to_numpy(), ci_level=ci_level)
    delta_ec_mean, delta_ec_lo, delta_ec_hi = bootstrap_ci(
        boot_df["delta_error_capture"].to_numpy(), ci_level=ci_level
    )
    delta_dp_mean, delta_dp_lo, delta_dp_hi = bootstrap_ci(
        boot_df["delta_deferral_precision"].to_numpy(), ci_level=ci_level
    )

    primary_risk_lower = bool(primary_hn["selective_risk"] < primary_h["selective_risk"])
    primary_ci_excludes_zero = bool(delta_risk_lo > 0)
    secondary_ec_higher = bool(primary_hn["error_capture"] > primary_h["error_capture"])
    secondary_precision_higher = bool(primary_hn["deferral_precision"] > primary_h["deferral_precision"])

    passed = bool(primary_risk_lower and primary_ci_excludes_zero)

    return {
        "primary_risk_HN_lower_than_H": primary_risk_lower,
        "primary_delta_risk_ci_excludes_zero": primary_ci_excludes_zero,
        "primary_risk_H": primary_h["selective_risk"],
        "primary_risk_HN": primary_hn["selective_risk"],
        "primary_delta_risk_mean": delta_risk_mean,
        "primary_delta_risk_ci_low": delta_risk_lo,
        "primary_delta_risk_ci_high": delta_risk_hi,
        "secondary_error_capture_HN_higher": secondary_ec_higher,
        "secondary_error_capture_H": primary_h["error_capture"],
        "secondary_error_capture_HN": primary_hn["error_capture"],
        "primary_delta_error_capture_mean": delta_ec_mean,
        "primary_delta_error_capture_ci_low": delta_ec_lo,
        "primary_delta_error_capture_ci_high": delta_ec_hi,
        "secondary_deferral_precision_HN_higher": secondary_precision_higher,
        "primary_delta_deferral_precision_mean": delta_dp_mean,
        "primary_delta_deferral_precision_ci_low": delta_dp_lo,
        "primary_delta_deferral_precision_ci_high": delta_dp_hi,
        "passed": passed,
        "hypothesis_supported": passed,
    }


def run_single_seed(
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: SelectivePredictionConfig,
    *,
    seed: int | None,
) -> dict[str, Any]:
    cal_sub = cal_df if seed is None else cal_df[cal_df["seed"] == seed].copy()
    test_sub = test_df if seed is None else test_df[test_df["seed"] == seed].copy()
    assert_no_leakage(cal_sub, test_sub)

    cal_scored, test_scored, combined_model = attach_failure_scores(cal_sub, test_sub, cfg)

    result_rows: list[dict[str, Any]] = []
    for defer_rate in cfg.secondary_deferral_rates:
        cov = 1.0 - defer_rate
        for scorer, col in [("H", "S_H"), ("H+N", "S_HN")]:
            m = evaluate_scorer_at_coverage(cal_scored, test_scored, score_col=col, target_coverage=cov, scorer=scorer)  # type: ignore[arg-type]
            m["seed"] = seed if seed is not None else "pooled"
            m["deferral_rate_target"] = defer_rate
            result_rows.append(m)

    curve_h = build_risk_coverage_from_calibration(cal_scored, test_scored, score_col="S_H", scorer="H")
    curve_hn = build_risk_coverage_from_calibration(cal_scored, test_scored, score_col="S_HN", scorer="H+N")
    curve_h["seed"] = seed if seed is not None else "pooled"
    curve_hn["seed"] = seed if seed is not None else "pooled"

    primary_h = evaluate_scorer_at_coverage(
        cal_scored, test_scored, score_col="S_H", target_coverage=cfg.primary_coverage, scorer="H"
    )
    primary_hn = evaluate_scorer_at_coverage(
        cal_scored, test_scored, score_col="S_HN", target_coverage=cfg.primary_coverage, scorer="H+N"
    )

    boot_df = bootstrap_paired_deltas(
        test_scored,
        threshold_h=primary_h["threshold"],
        threshold_hn=primary_hn["threshold"],
        cfg=cfg,
    )
    gate = preregistered_verdict(primary_h, primary_hn, boot_df, ci_level=cfg.ci_level)

    per_sample = test_scored.copy()
    per_sample["deferred_H"] = (per_sample["S_H"] > primary_h["threshold"]).astype(int)
    per_sample["deferred_HN"] = (per_sample["S_HN"] > primary_hn["threshold"]).astype(int)

    return {
        "seed": seed,
        "results_rows": result_rows,
        "curve_h": curve_h,
        "curve_hn": curve_hn,
        "primary_h": primary_h,
        "primary_hn": primary_hn,
        "bootstrap_df": boot_df,
        "gate": gate,
        "per_sample": per_sample,
        "combined_model": combined_model,
        "cal_thresholds": {"H": primary_h["threshold"], "H+N": primary_hn["threshold"]},
    }


def plot_selective_prediction_figure(
    curve_h: pd.DataFrame,
    curve_hn: pd.DataFrame,
    primary_h: dict[str, Any],
    primary_hn: dict[str, Any],
    out_path: Path,
    *,
    primary_deferral_rate: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(curve_h["coverage"], curve_h["selective_risk"], "o-", label="H", color="steelblue", linewidth=2)
    ax.plot(curve_hn["coverage"], curve_hn["selective_risk"], "s-", label="H+N", color="crimson", linewidth=2)
    ax.axvline(1.0 - primary_deferral_rate, color="gray", linestyle="--", alpha=0.7, label="primary deferral")
    ax.set_xlabel("Coverage (fraction retained)")
    ax.set_ylabel("Selective risk (error rate retained)")
    ax.set_title("Risk-coverage curves (calibrated thresholds)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    labels = ["Retained\n(H)", "Deferred\n(H)", "Retained\n(H+N)", "Deferred\n(H+N)"]
    rates = [
        primary_h["selective_risk"],
        primary_h["error_rate_deferred"],
        primary_hn["selective_risk"],
        primary_hn["error_rate_deferred"],
    ]
    colors = ["steelblue", "lightsteelblue", "crimson", "lightsalmon"]
    ax.bar(labels, rates, color=colors)
    ax.set_ylabel("Error rate")
    ax.set_title(f"Retained vs deferred error rates ({int(primary_deferral_rate*100)}% deferral)")
    ax.set_ylim(0, max(r for r in rates if np.isfinite(r)) * 1.15)

    ax = axes[2]
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width / 2, [primary_h["error_capture"], primary_h["deferral_precision"]], width, label="H", color="steelblue")
    ax.bar(x + width / 2, [primary_hn["error_capture"], primary_hn["deferral_precision"]], width, label="H+N", color="crimson")
    ax.set_xticks(x)
    ax.set_xticklabels(["Error capture", "Deferral precision"])
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Error capture & precision ({int(primary_deferral_rate*100)}% deferral)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _generate_summary_md(
    cfg: SelectivePredictionConfig,
    per_seed_df: pd.DataFrame,
    pooled_gate: dict[str, Any],
    primary_h: dict[str, Any],
    primary_hn: dict[str, Any],
) -> str:
    verdict = "PASS" if pooled_gate["passed"] else "FAIL"
    lines = [
        "# Selective Prediction / Human Deferral — Summary",
        "",
        "## Research claim",
        "",
        "> Can optimization-history novelty N improve selective prediction by identifying",
        "> samples on which the model is likely to be wrong, so those samples can be deferred to a human?",
        "",
        f"## PRE-REGISTERED VERDICT: **{verdict}**",
        "",
        f"Hypothesis supported: **{pooled_gate['hypothesis_supported']}**",
        "",
        "## Primary results (pooled external, ~20% deferral / 80% coverage)",
        "",
        "| Scorer | Selective risk | Error capture | Deferral precision | Coverage |",
        "|--------|----------------|---------------|----------------------|----------|",
        f"| H | {primary_h['selective_risk']:.4f} | {primary_h['error_capture']:.4f} | {primary_h['deferral_precision']:.4f} | {primary_h['coverage']:.4f} |",
        f"| H+N | {primary_hn['selective_risk']:.4f} | {primary_hn['error_capture']:.4f} | {primary_hn['deferral_precision']:.4f} | {primary_hn['coverage']:.4f} |",
        "",
        f"- delta_risk (Risk_H - Risk_HN): **{pooled_gate['primary_delta_risk_mean']:.4f}**",
        f"  - 95% CI: [{pooled_gate['primary_delta_risk_ci_low']:.4f}, {pooled_gate['primary_delta_risk_ci_high']:.4f}]",
        f"- delta_error_capture (EC_HN - EC_H): **{pooled_gate['primary_delta_error_capture_mean']:.4f}**",
        f"  - 95% CI: [{pooled_gate['primary_delta_error_capture_ci_low']:.4f}, {pooled_gate['primary_delta_error_capture_ci_high']:.4f}]",
        "",
        "## Per-seed primary metrics (20% deferral)",
        "",
        "```",
        per_seed_df.to_string(index=False),
        "```",
        "",
        "## Interpretation",
        "",
    ]
    if pooled_gate["passed"]:
        lines.append(
            "**PASS:** At matched ~20% deferral, H+N achieves lower selective risk than H on external "
            "DermaMNIST-E, with bootstrap CI excluding zero. H+N preferentially identifies model failures "
            "for human deferral. This does **not** claim human-level performance — only that deferred "
            "samples contain a higher fraction of model errors than entropy alone."
        )
    else:
        lines.append(
            "**FAIL:** Pre-registered selective-risk criterion not met at ~20% deferral on external test."
        )
    lines.extend(["", "## Protocol", "", f"- Calibration: ID split only (threshold selection)", f"- Test: {cfg.evaluation_domain} (frozen threshold, single application)", f"- Seeds: {list(cfg.seeds)}", ""])
    return "\n".join(lines)


def run_selective_prediction_experiment(cfg: SelectivePredictionConfig) -> dict[str, Any]:
    cfg = cfg.resolve_paths()
    output_dir = cfg.output_dir
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "audit").mkdir(exist_ok=True)

    cal_all = load_calibration_frames(cfg)
    test_all = load_test_frames(cfg)

    per_seed_runs: dict[str, Any] = {}
    all_result_rows: list[dict[str, Any]] = []
    per_seed_primary_rows: list[dict[str, Any]] = []
    all_boot_rows: list[pd.DataFrame] = []

    for seed in cfg.seeds:
        run = run_single_seed(cal_all, test_all, cfg, seed=seed)
        per_seed_runs[str(seed)] = run
        all_result_rows.extend(run["results_rows"])
        for scorer, primary in [("H", run["primary_h"]), ("H+N", run["primary_hn"])]:
            row = {k: v for k, v in primary.items()}
            row["seed"] = seed
            row["scorer"] = scorer
            row["deferral_rate_target"] = PRIMARY_DEFERRAL_RATE
            per_seed_primary_rows.append(row)
        boot = run["bootstrap_df"].copy()
        boot["seed"] = seed
        all_boot_rows.append(boot)

    pooled_run = run_single_seed(cal_all, test_all, cfg, seed=None)
    all_result_rows.extend(pooled_run["results_rows"])
    for scorer, primary in [("H", pooled_run["primary_h"]), ("H+N", pooled_run["primary_hn"])]:
        row = {k: v for k, v in primary.items()}
        row["seed"] = "pooled"
        row["scorer"] = scorer
        row["deferral_rate_target"] = PRIMARY_DEFERRAL_RATE
        per_seed_primary_rows.append(row)
    pooled_boot = pooled_run["bootstrap_df"].copy()
    pooled_boot["seed"] = "pooled"
    all_boot_rows.append(pooled_boot)

    results_df = pd.DataFrame(all_result_rows)
    per_seed_df = pd.DataFrame(per_seed_primary_rows)
    bootstrap_df = pd.concat(all_boot_rows, ignore_index=True)
    gate = pooled_run["gate"]

    if cfg.save_artifacts:
        config_payload = {
            **{k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
            "seeds": list(cfg.seeds),
            "h_definition": "normalized_entropy = predictive_entropy / log(num_classes); S_H = H",
            "hn_definition": "S_HN = logistic(H,N) fit on calibration only (existing clinical_failure scorer)",
            "threshold_protocol": "calibrate on ID calibration; defer score > threshold; freeze for external test",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "platform": platform.platform(),
        }
        (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
        (output_dir / "audit" / "leakage_assertions.json").write_text(
            json.dumps({"calibration_test_disjoint": True, "threshold_fit_on_calibration_only": True}, indent=2),
            encoding="utf-8",
        )
        (output_dir / "audit" / "frozen_thresholds.json").write_text(
            json.dumps(
                {
                    str(s): per_seed_runs[str(s)]["cal_thresholds"]
                    for s in cfg.seeds
                }
                | {"pooled": pooled_run["cal_thresholds"]},
                indent=2,
            ),
            encoding="utf-8",
        )

        results_df.to_csv(output_dir / "selective_prediction_results.csv", index=False)
        per_seed_df.to_csv(output_dir / "selective_prediction_per_seed.csv", index=False)
        bootstrap_df.to_csv(output_dir / "selective_prediction_bootstrap.csv", index=False)
        pooled_run["per_sample"].to_csv(output_dir / "selective_prediction_per_sample.csv", index=False)
        (output_dir / "decision_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

        summary = _generate_summary_md(
            cfg, per_seed_df[per_seed_df["seed"] != "pooled"], gate,
            pooled_run["primary_h"], pooled_run["primary_hn"],
        )
        (output_dir / "selective_prediction_summary.md").write_text(summary, encoding="utf-8")

        if cfg.show_plots:
            plot_selective_prediction_figure(
                pooled_run["curve_h"],
                pooled_run["curve_hn"],
                pooled_run["primary_h"],
                pooled_run["primary_hn"],
                output_dir / "figures" / "selective_prediction_summary.png",
                primary_deferral_rate=PRIMARY_DEFERRAL_RATE,
            )

    return {
        "config": cfg,
        "results_df": results_df,
        "per_seed_df": per_seed_df,
        "bootstrap_df": bootstrap_df,
        "decision_gate": gate,
        "per_seed_runs": per_seed_runs,
        "pooled_run": pooled_run,
        "output_dir": output_dir,
    }


def print_final_verdict(results: dict[str, Any]) -> None:
    gate = results["decision_gate"]
    pooled = results["pooled_run"]
    h, hn = pooled["primary_h"], pooled["primary_hn"]
    print("=" * 72)
    print("SELECTIVE PREDICTION / HUMAN DEFERRAL — PRE-REGISTERED VERDICT")
    print("=" * 72)
    print(f"  VERDICT:                       {'PASS' if gate['passed'] else 'FAIL'}")
    print(f"  Hypothesis supported:          {gate['hypothesis_supported']}")
    print("-" * 72)
    print(f"  Primary (~20% deferral, external test, pooled):")
    print(f"    Selective risk H:            {h['selective_risk']:.4f}")
    print(f"    Selective risk H+N:          {hn['selective_risk']:.4f}")
    print(f"    Risk_HN < Risk_H:            {gate['primary_risk_HN_lower_than_H']}")
    print(f"    delta_risk = Risk_H - Risk_HN: {gate['primary_delta_risk_mean']:.4f}")
    print(f"    95% CI: [{gate['primary_delta_risk_ci_low']:.4f}, {gate['primary_delta_risk_ci_high']:.4f}]")
    print(f"    CI excludes 0:               {gate['primary_delta_risk_ci_excludes_zero']}")
    print("  Secondary:")
    print(f"    ErrorCapture H:              {h['error_capture']:.4f}")
    print(f"    ErrorCapture H+N:            {hn['error_capture']:.4f}")
    print(f"    EC_HN > EC_H:                {gate['secondary_error_capture_HN_higher']}")
    print(f"    DeferralPrecision H:         {h['deferral_precision']:.4f}")
    print(f"    DeferralPrecision H+N:       {hn['deferral_precision']:.4f}")
    print(f"    Prec_HN > Prec_H:            {gate['secondary_deferral_precision_HN_higher']}")
    print("=" * 72)
    if gate["passed"]:
        print(
            "SUPPORTS: Optimization-history novelty improves selective prediction by "
            "preferentially identifying samples on which the model is likely to fail."
        )
    else:
        print("DOES NOT SUPPORT the selective-prediction claim under pre-registered criteria.")
    print("=" * 72)
