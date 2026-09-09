"""End-to-end orchestration for the final clinical failure detection experiment."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.common.clinical_corruption import corruption_config_dict
from research.common.clinical_log import clinical_log
from research.common.clinical_datasets import (
    ClinicalDatasetConfig,
    ClinicalTask,
    build_dataset_manifest,
    load_clinical_bundle,
    save_dataset_manifest,
)
from research.common.clinical_failure import (
    FinalClinicalConfig,
    aggregate_metrics_table,
    attach_scores_from_calibration,
    bootstrap_grouped,
    build_all_population_frames,
    confident_wrong_analysis,
    deferral_metrics_at_coverage,
    ensure_output_layout,
    plot_architecture_diagram,
    plot_calibration,
    plot_confident_wrong,
    plot_risk_coverage,
    plot_roc_external,
    preregistration_decision,
    risk_coverage_curve,
    run_leakage_checks,
    write_final_readme,
)
from research.common.clinical_training import (
    ClinicalTrainingConfig,
    build_model_manifest,
    checkpoint_path,
    memory_path,
    save_checkpoint,
    save_frozen_memory,
    train_one_seed,
    evaluate_split,
    load_checkpoint,
)


def scored_cache_path(dirs: dict[str, Path], task: ClinicalTask, seed: int) -> Path:
    return dirs["scored"] / f"{task}_seed{seed}.csv"


def summarize_clinical_artifacts(cfg: FinalClinicalConfig) -> dict[str, Any]:
    """Report which on-disk artifacts exist for each task/seed and final exports."""
    cfg = cfg.resolve_paths()
    dirs = ensure_output_layout(Path(cfg.output_dir))
    per_run: list[dict[str, Any]] = []
    for task in cfg.datasets:
        for seed in cfg.seeds:
            per_run.append(
                {
                    "task": task,
                    "seed": seed,
                    "checkpoint": checkpoint_path(dirs["root"], task, seed).exists(),
                    "memory": memory_path(dirs["root"], task, seed).exists(),
                    "calibration_csv": (dirs["calibration"] / f"{task}_seed{seed}.csv").exists(),
                    "scored_csv": scored_cache_path(dirs, task, seed).exists(),
                }
            )
    per_run_df = pd.DataFrame(per_run)
    final_exports = {
        "aggregate_metrics_csv": (dirs["metrics"] / "aggregate_metrics.csv").exists(),
        "bootstrap_results_csv": (dirs["statistics"] / "bootstrap_results.csv").exists(),
        "confident_wrong_csv": (dirs["metrics"] / "confident_wrong.csv").exists(),
        "decision_gate_json": (dirs["metrics"] / "decision_gate.json").exists(),
        "figure_count": len(list(dirs["figures"].glob("*.png"))),
    }
    scoring_complete = bool(len(per_run_df) and per_run_df["scored_csv"].all())
    final_exports_exist = bool(
        final_exports["aggregate_metrics_csv"]
        and final_exports["bootstrap_results_csv"]
        and final_exports["confident_wrong_csv"]
        and final_exports["figure_count"] > 0
    )
    ready_for_display = bool(scoring_complete and final_exports_exist)
    return {
        "output_dir": dirs["root"],
        "per_run": per_run_df,
        "final_exports": final_exports,
        "scoring_complete": scoring_complete,
        "final_exports_exist": final_exports_exist,
        "ready_for_display": ready_for_display,
    }


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def load_scored_results_from_disk(cfg: FinalClinicalConfig) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Load per-run scored CSV caches written during the scoring phase."""
    cfg = cfg.resolve_paths()
    dirs = ensure_output_layout(Path(cfg.output_dir))
    scored_parts: list[pd.DataFrame] = []
    cal_parts: list[pd.DataFrame] = []
    missing: list[str] = []
    for task in cfg.datasets:
        for seed in cfg.seeds:
            score_cache = scored_cache_path(dirs, task, seed)
            cal_csv = dirs["calibration"] / f"{task}_seed{seed}.csv"
            if not score_cache.exists():
                missing.append(str(score_cache))
                continue
            scored_parts.append(pd.read_csv(score_cache))
            if cal_csv.exists():
                cal_parts.append(pd.read_csv(cal_csv))
    if missing:
        raise FileNotFoundError(
            "Missing scored caches required for finalize:\n  " + "\n  ".join(missing)
        )
    return pd.concat(scored_parts, ignore_index=True), cal_parts


def _load_model_summaries(cfg: FinalClinicalConfig, dirs: dict[str, Path]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for task in cfg.datasets:
        for seed in cfg.seeds:
            ckpt = checkpoint_path(dirs["root"], task, seed)
            if not ckpt.exists():
                raise FileNotFoundError(f"Missing checkpoint required for finalize: {ckpt}")
            payload = load_checkpoint(ckpt)
            summaries.append(payload.get("summary", {"task": task, "seed": seed}))
    return summaries


def _export_clinical_results(
    cfg: FinalClinicalConfig,
    *,
    dirs: dict[str, Path],
    bundles: dict[ClinicalTask, Any],
    scored_all: pd.DataFrame,
    cal_parts: list[pd.DataFrame],
    model_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregation, bootstrap, leakage checks, figures, and exports from scored data."""
    scored_all = scored_all.copy()
    scored_all["patient_id"] = ""
    scored_all["true_label"] = scored_all["label"]
    scored_all["predicted_label"] = scored_all["prediction"]

    metric_rows: list[dict[str, Any]] = []
    for scorer, col in (("H", "S_H"), ("N", "S_N"), ("H+N", "S_HN")):
        part = scored_all.copy()
        part["scorer"] = scorer
        part["score"] = part[col]
        metric_rows.append(part)
    metrics_long = pd.concat(metric_rows, ignore_index=True)

    aggregate_df = aggregate_metrics_table(metrics_long, target_coverage=cfg.target_coverage)
    aggregate_df.to_csv(dirs["metrics"] / "aggregate_metrics.csv", index=False)

    export_cols = [
        "dataset",
        "task",
        "seed",
        "sample_id",
        "patient_id",
        "domain",
        "corruption",
        "true_label",
        "predicted_label",
        "correct",
        "max_probability",
        "predictive_entropy",
        "normalized_entropy",
        "memory_novelty",
        "memory_disagreement",
        "combined_failure_probability",
        "deferred_at_80pct",
        "gradient_norm",
        "memory_norms",
    ]
    for col in export_cols:
        if col not in scored_all.columns:
            scored_all[col] = np.nan
    scored_all[export_cols].to_csv(dirs["metrics"] / "per_sample_metrics.csv", index=False)

    rc_rows: list[pd.DataFrame] = []
    for (task, domain, seed, scorer), g in metrics_long.groupby(["task", "domain", "seed", "scorer"]):
        curve = risk_coverage_curve(g["error"].to_numpy(), g["score"].to_numpy())
        curve["task"] = task
        curve["domain"] = domain
        curve["seed"] = seed
        curve["scorer"] = scorer
        rc_rows.append(curve)
    risk_coverage_df = pd.concat(rc_rows, ignore_index=True)
    risk_coverage_df.to_csv(dirs["metrics"] / "risk_coverage.csv", index=False)

    clinical_log(f"Bootstrap analysis ({cfg.bootstrap_replicates} replicates) …", verbose=cfg.verbose)
    bootstrap_rows: list[dict[str, Any]] = []
    for task in cfg.datasets:
        for domain in ("id", "external"):
            for seed in cfg.seeds:
                base = scored_all[
                    (scored_all["task"] == task)
                    & (scored_all["domain"] == domain)
                    & (scored_all["seed"] == seed)
                ].copy()
                if base.empty:
                    continue
                base["error"] = (1 - base["correct"].astype(int)).astype(int)
                boot_auroc = bootstrap_grouped(
                    base,
                    score_a="S_H",
                    score_b="S_HN",
                    group_col="sample_id",
                    n_bootstrap=cfg.bootstrap_replicates,
                    seed=seed,
                    metric="auroc",
                )
                boot_aurc = bootstrap_grouped(
                    base,
                    score_a="S_H",
                    score_b="S_HN",
                    group_col="sample_id",
                    n_bootstrap=cfg.bootstrap_replicates,
                    seed=seed + 1,
                    metric="aurc",
                    target_coverage=cfg.target_coverage,
                )
                bootstrap_rows.append(
                    {
                        "task": task,
                        "domain": domain,
                        "seed": seed,
                        "delta_auroc_mean": boot_auroc["delta_mean"],
                        "delta_auroc_ci_low": boot_auroc["delta_ci_low"],
                        "delta_auroc_ci_high": boot_auroc["delta_ci_high"],
                        "delta_auroc_p": boot_auroc["p_value"],
                        "delta_aurc_mean": boot_aurc["delta_mean"],
                        "delta_aurc_ci_low": boot_aurc["delta_ci_low"],
                        "delta_aurc_ci_high": boot_aurc["delta_ci_high"],
                    }
                )
                clinical_log(
                    f"  bootstrap {task}/{domain}/seed{seed}: "
                    f"ΔAUROC={boot_auroc['delta_mean']:.4f} ΔAURC={boot_aurc['delta_mean']:.4f}",
                    verbose=cfg.verbose,
                    level=2,
                )
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_df.to_csv(dirs["statistics"] / "bootstrap_results.csv", index=False)
    bootstrap_df.to_csv(dirs["statistics"] / "auroc_comparisons.csv", index=False)
    bootstrap_df[["task", "domain", "seed", "delta_aurc_mean", "delta_aurc_ci_low", "delta_aurc_ci_high"]].to_csv(
        dirs["statistics"] / "aurc_comparisons.csv",
        index=False,
    )
    bootstrap_df[
        ["task", "domain", "seed", "delta_auroc_ci_low", "delta_auroc_ci_high", "delta_aurc_ci_low", "delta_aurc_ci_high"]
    ].to_csv(dirs["statistics"] / "confidence_intervals.csv", index=False)

    seed_agg = (
        bootstrap_df.groupby(["task", "domain"])
        .agg(
            delta_auroc_mean=("delta_auroc_mean", "mean"),
            delta_auroc_std=("delta_auroc_mean", "std"),
            delta_aurc_mean=("delta_aurc_mean", "mean"),
            delta_aurc_std=("delta_aurc_mean", "std"),
        )
        .reset_index()
    )
    seed_agg.to_csv(dirs["statistics"] / "seed_aggregation.csv", index=False)

    cw_rows: list[dict[str, Any]] = []
    for task in cfg.datasets:
        for seed in cfg.seeds:
            sub = scored_all[(scored_all["task"] == task) & (scored_all["seed"] == seed) & (scored_all["domain"] == "id")]
            stats = confident_wrong_analysis(sub)
            stats.update({"task": task, "seed": seed})
            cw_rows.append(stats)
    confident_wrong_df = pd.DataFrame(cw_rows)
    confident_wrong_df.to_csv(dirs["metrics"] / "confident_wrong.csv", index=False)

    cal_metrics = aggregate_df[aggregate_df["Scorer"] == "H+N"][["Dataset", "Shift", "Brier", "ECE"]]
    cal_metrics.to_csv(dirs["metrics"] / "calibration.csv", index=False)

    cal_ids = set(pd.concat(cal_parts)["sample_id"].astype(str)) if cal_parts else set()
    test_ids = set()
    for task, bundle in bundles.items():
        test_ids.update(bundle.sample_ids["test"].astype(str))
    leakage = run_leakage_checks(
        bundles=bundles,
        cal_sample_ids=cal_ids,
        test_sample_ids=test_ids,
        external_used_in_training=False,
        memory_updated_at_eval=False,
        model_updated_at_eval=False,
        logistic_fit_on_test=False,
        corruption_tuned_on_test=False,
        threshold_from_test_labels=False,
        patient_leakage=False,
    )
    (dirs["audit"] / "leakage_checks.json").write_text(json.dumps(leakage, indent=2), encoding="utf-8")

    model_manifest = build_model_manifest(model_summaries)
    (dirs["audit"] / "model_manifest.json").write_text(json.dumps(model_manifest, indent=2, default=_json_default), encoding="utf-8")
    (dirs["audit"] / "experiment_hash.json").write_text(
        json.dumps({"created_unix": time.time(), "n_samples": len(scored_all)}, indent=2),
        encoding="utf-8",
    )

    plot_architecture_diagram(dirs["figures"] / "architecture.png")
    for task in cfg.datasets:
        plot_roc_external(scored_all, task, dirs["figures"] / f"roc_{task}_external.png")
        plot_risk_coverage(scored_all, task, "external", dirs["figures"] / f"risk_coverage_{task}_external.png")
        sub_id = scored_all[(scored_all["task"] == task) & (scored_all["domain"] == "id")]
        if len(sub_id):
            plot_confident_wrong(sub_id, task, dirs["figures"] / f"confident_wrong_{task}.png")
            plot_calibration(sub_id, dirs["figures"] / f"calibration_{task}.png")

    for task in cfg.datasets:
        task_df = scored_all[scored_all["task"] == task]
        task_df[task_df["domain"] == "id"].to_csv(dirs["predictions"] / f"{task}_id.csv", index=False)
        task_df[task_df["domain"] == "external"].to_csv(dirs["predictions"] / f"{task}_external.csv", index=False)

    gate = preregistration_decision(bootstrap_df, confident_wrong_df, seed_agg)
    (dirs["metrics"] / "decision_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    write_final_readme(dirs["root"] / "README.md", cfg, gate, aggregate_df)
    clinical_log(f"Done. Hypothesis supported: {gate['hypothesis_supported']}", verbose=cfg.verbose)

    (dirs["root"] / "environment.txt").write_text(
        "\n".join([f"{name}=={ver}" for name, ver in sorted(_installed_versions().items())]),
        encoding="utf-8",
    )
    (dirs["root"] / "seeds.json").write_text(json.dumps({"seeds": list(cfg.seeds)}, indent=2), encoding="utf-8")

    return {
        "config": cfg,
        "bundles": bundles,
        "scored_all": scored_all,
        "aggregate_metrics": aggregate_df,
        "bootstrap": bootstrap_df,
        "confident_wrong": confident_wrong_df,
        "leakage_checks": leakage,
        "decision_gate": gate,
        "output_dir": dirs["root"],
    }


def finalize_clinical_experiment(cfg: FinalClinicalConfig) -> dict[str, Any]:
    """Complete analysis/exports from saved scored/ caches — no training or MSA scoring."""
    cfg = cfg.resolve_paths()
    dirs = ensure_output_layout(Path(cfg.output_dir))
    clinical_log(f"Finalizing from scored caches in {dirs['scored']}", verbose=cfg.verbose)

    scored_all, cal_parts = load_scored_results_from_disk(cfg)
    model_summaries = _load_model_summaries(cfg, dirs)

    bundles: dict[ClinicalTask, Any] = {}
    for task in cfg.datasets:
        ds_cfg = ClinicalDatasetConfig(
            task=task,
            data_dir=Path(cfg.data_dir),
            max_train=cfg.max_train,
            max_cal=cfg.max_cal,
            max_test=cfg.max_test,
            max_external=cfg.max_external,
        )
        bundles[task] = load_clinical_bundle(ds_cfg)

    return _export_clinical_results(
        cfg,
        dirs=dirs,
        bundles=bundles,
        scored_all=scored_all,
        cal_parts=cal_parts,
        model_summaries=model_summaries,
    )


def load_clinical_results(cfg: FinalClinicalConfig) -> dict[str, Any]:
    """Load completed experiment outputs from disk without recomputation."""
    cfg = cfg.resolve_paths()
    dirs = ensure_output_layout(Path(cfg.output_dir))
    metrics_dir = dirs["metrics"]
    stats_dir = dirs["statistics"]
    leakage_path = dirs["audit"] / "leakage_checks.json"
    gate_path = metrics_dir / "decision_gate.json"
    for path in (
        metrics_dir / "aggregate_metrics.csv",
        stats_dir / "bootstrap_results.csv",
        metrics_dir / "confident_wrong.csv",
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing completed result artifact: {path}")
    leakage = json.loads(leakage_path.read_text(encoding="utf-8")) if leakage_path.exists() else {}
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else {}
    scored_all, _ = load_scored_results_from_disk(cfg)
    datasets = set(cfg.datasets)
    aggregate_metrics = pd.read_csv(metrics_dir / "aggregate_metrics.csv")
    aggregate_metrics = aggregate_metrics[aggregate_metrics["Dataset"].isin(datasets)].copy()
    bootstrap = pd.read_csv(stats_dir / "bootstrap_results.csv")
    bootstrap = bootstrap[bootstrap["task"].isin(datasets)].copy()
    confident_wrong = pd.read_csv(metrics_dir / "confident_wrong.csv")
    confident_wrong = confident_wrong[confident_wrong["task"].isin(datasets)].copy()
    return {
        "config": cfg,
        "bundles": {},
        "scored_all": scored_all,
        "aggregate_metrics": aggregate_metrics,
        "bootstrap": bootstrap,
        "confident_wrong": confident_wrong,
        "leakage_checks": leakage,
        "decision_gate": gate,
        "output_dir": dirs["root"],
    }


def run_clinical_experiment(cfg: FinalClinicalConfig) -> dict[str, Any]:
    """Run full pipeline, finalize from caches, or load completed outputs as appropriate."""
    cfg = cfg.resolve_paths()
    status = summarize_clinical_artifacts(cfg)
    if status["ready_for_display"]:
        clinical_log("Loading completed results from disk", verbose=cfg.verbose)
        return load_clinical_results(cfg)
    if status["scoring_complete"]:
        clinical_log("Scored caches found — finalizing without MSA scoring", verbose=cfg.verbose)
        return finalize_clinical_experiment(cfg)
    clinical_log(
        f"Running train/score pipeline for {cfg.datasets} "
        f"(missing scored caches: {int((~status['per_run']['scored_csv']).sum())} runs)",
        verbose=cfg.verbose,
    )
    return run_final_clinical_experiment(cfg)


def run_final_clinical_experiment(cfg: FinalClinicalConfig) -> dict[str, Any]:
    cfg = cfg.resolve_paths()
    dirs = ensure_output_layout(Path(cfg.output_dir))
    train_cfg = ClinicalTrainingConfig(
        seeds=cfg.seeds,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        output_dir=Path(cfg.output_dir),
        verbose=cfg.verbose,
    )

    clinical_log(f"Output directory: {cfg.output_dir}", verbose=cfg.verbose)
    clinical_log(f"Datasets: {cfg.datasets} | Seeds: {cfg.seeds}", verbose=cfg.verbose)

    config_payload = {**asdict(cfg), "corruption": corruption_config_dict()}
    for key in ("repo_root", "data_dir", "output_dir"):
        if config_payload.get(key) is not None:
            config_payload[key] = str(config_payload[key])
    (dirs["root"] / "config.json").write_text(
        json.dumps(config_payload, indent=2, default=_json_default),
        encoding="utf-8",
    )

    bundles: dict[ClinicalTask, Any] = {}
    for task in cfg.datasets:
        clinical_log(f"Loading dataset: {task} …", verbose=cfg.verbose)
        ds_cfg = ClinicalDatasetConfig(
            task=task,
            data_dir=Path(cfg.data_dir),
            max_train=cfg.max_train,
            max_cal=cfg.max_cal,
            max_test=cfg.max_test,
            max_external=cfg.max_external,
        )
        bundles[task] = load_clinical_bundle(ds_cfg)
        b = bundles[task]
        clinical_log(
            f"  {task}: train={len(b.x_train)} cal={len(b.x_cal)} test={len(b.x_test)} "
            f"external={len(b.x_external)} (proxy={b.external_proxy_used})",
            verbose=cfg.verbose,
        )

    manifest = build_dataset_manifest(bundles, data_dir=Path(cfg.data_dir))
    save_dataset_manifest(manifest, dirs["audit"] / "dataset_manifest.json")

    model_summaries: list[dict[str, Any]] = []
    memory_summaries: list[dict[str, Any]] = []
    all_scored_parts: list[pd.DataFrame] = []
    cal_parts: list[pd.DataFrame] = []

    for task in cfg.datasets:
        bundle = bundles[task]
        for seed in cfg.seeds:
            ckpt = checkpoint_path(dirs["root"], task, seed)
            if cfg.run_training_if_missing and not ckpt.exists():
                clinical_log(f"Training {task} seed={seed} …", verbose=cfg.verbose)
                params, opt_state, _per_sample, _metrics, summary = train_one_seed(
                    bundle,
                    seed=seed,
                    cfg=train_cfg,
                )
                save_checkpoint(
                    ckpt,
                    params=params,
                    opt_state=opt_state,
                    task=task,
                    seed=seed,
                    cfg=train_cfg,
                    summary=summary,
                )
                mem_summary = save_frozen_memory(memory_path(dirs["root"], task, seed), opt_state, task=task, seed=seed)
                memory_summaries.append({"task": task, "seed": seed, **mem_summary})
                model_summaries.append(summary)
                clinical_log(
                    f"  saved checkpoint + memory | test_acc={summary['test_acc']:.3f} "
                    f"({summary['train_time_s']:.1f}s)",
                    verbose=cfg.verbose,
                )
            elif ckpt.exists():
                clinical_log(f"Using existing checkpoint: {task} seed={seed}", verbose=cfg.verbose)
                payload = load_checkpoint(ckpt)
                model_summaries.append(payload.get("summary", {"task": task, "seed": seed}))
            else:
                raise FileNotFoundError(f"Missing checkpoint {ckpt}; set run_training_if_missing=True")

            score_cache = scored_cache_path(dirs, task, seed)
            cal_csv = dirs["calibration"] / f"{task}_seed{seed}.csv"
            if cfg.resume_scoring and score_cache.exists():
                clinical_log(f"Using cached scores: {task} seed={seed}", verbose=cfg.verbose)
                scored = pd.read_csv(score_cache)
                if cal_csv.exists():
                    cal_df = pd.read_csv(cal_csv)
                else:
                    payload = load_checkpoint(ckpt)
                    cal_df = evaluate_split(
                        payload["params"],
                        bundle.x_cal,
                        bundle.y_cal,
                        bundle.sample_ids["cal"],
                        payload["opt_state"],
                        seed=seed,
                        epoch=train_cfg.epochs - 1,
                        step=-1,
                        split="calibration",
                        cfg=train_cfg,
                        score_msa=True,
                    )
                    cal_df["task"] = task
                    cal_df["dataset"] = task
                    cal_df["domain"] = "calibration"
                    cal_df["seed"] = seed
                    cal_df.to_csv(cal_csv, index=False)
                cal_parts.append(cal_df)
                all_scored_parts.append(scored)
                clinical_log(
                    f"  loaded {len(scored)} cached samples | "
                    f"ID err={scored.loc[scored['domain']=='id','error'].mean():.3f}",
                    verbose=cfg.verbose,
                )
                continue

            payload = load_checkpoint(ckpt)
            params = payload["params"]
            opt_state = payload["opt_state"]

            clinical_log(f"Scoring populations: {task} seed={seed} …", verbose=cfg.verbose)

            cal_df = evaluate_split(
                params,
                bundle.x_cal,
                bundle.y_cal,
                bundle.sample_ids["cal"],
                opt_state,
                seed=seed,
                epoch=train_cfg.epochs - 1,
                step=-1,
                split="calibration",
                cfg=train_cfg,
                score_msa=True,
            )
            cal_df["task"] = task
            cal_df["dataset"] = task
            cal_df["domain"] = "calibration"
            cal_df["seed"] = seed
            cal_parts.append(cal_df)

            pop_df = build_all_population_frames(
                bundle,
                task=task,
                seed=seed,
                train_cfg=train_cfg,
                checkpoint_path=ckpt,
                corruption_base_seed=seed * 1000 + hash(task) % 1000,
                verbose=cfg.verbose,
            )
            pop_df["seed"] = seed
            scored, combined = attach_scores_from_calibration(
                cal_df,
                pop_df,
                num_classes=bundle.num_classes,
            )
            defer = deferral_metrics_at_coverage(
                scored["error"].to_numpy(),
                scored["S_HN"].to_numpy(),
                target_coverage=cfg.target_coverage,
            )
            scored["deferred_at_80pct"] = 0
            order = np.argsort(scored["S_HN"].to_numpy())
            k = max(1, int(round(cfg.target_coverage * len(scored))))
            scored.loc[scored.index[order[k:]], "deferred_at_80pct"] = 1
            all_scored_parts.append(scored)
            clinical_log(
                f"  scored {len(scored)} samples | ID err={scored.loc[scored['domain']=='id','error'].mean():.3f}",
                verbose=cfg.verbose,
            )

            cal_df.to_csv(cal_csv, index=False)
            if cfg.save_artifacts:
                scored.to_csv(score_cache, index=False)

    return _export_clinical_results(
        cfg,
        dirs=dirs,
        bundles=bundles,
        scored_all=pd.concat(all_scored_parts, ignore_index=True),
        cal_parts=cal_parts,
        model_summaries=model_summaries,
    )


def _installed_versions() -> dict[str, str]:
    import importlib.metadata as md

    names = ["jax", "optax", "numpy", "pandas", "scikit-learn", "matplotlib", "scipy"]
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = "not_installed"
    out["python"] = sys.version.split()[0]
    return out
