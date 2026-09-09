#!/usr/bin/env python
"""Run label-free deployment pipeline and offline memory-information evaluation.

Example:
    python research/associative_memory_fft/run_deployment_pipeline.py \\
        --task dermamnist --seed 42 \\
        --artifacts-dir research/associative_memory_fft/artifacts \\
        --output-dir research/associative_memory_fft/deployment/dermamnist_seed42
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "research").is_dir() and (parent / "optimizer").is_dir():
            return parent
    return Path.cwd()


def main() -> int:
    parser = argparse.ArgumentParser(description="Historical associative memory deployment pipeline")
    parser.add_argument("--task", default="dermamnist")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("research/associative_memory_fft/artifacts"),
    )
    parser.add_argument("--output-dir", type=Path, required=False)
    parser.add_argument("--data-dir", type=Path, default=Path("data/clinical"))
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["cal", "external"],
        choices=["train", "cal", "test", "external"],
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Subsample each split for smoke runs")
    parser.add_argument("--include-history", action="store_true", default=True)
    parser.add_argument("--no-history", action="store_true", help="Skip checkpoint trajectory in records")
    args = parser.parse_args()

    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from research.common.clinical_datasets import ClinicalDatasetConfig, load_clinical_bundle
    from research.common.clinical_training import checkpoint_path
    from research.common.deployment_pipeline import (
        offline_evaluate_memory_information,
        plot_deployment_diagnostics,
        run_deployment_on_split,
        run_deployment_sanity_checks,
        save_deployment_records,
        summarize_deployment_diagnostics,
    )
    from research.common.memory import load_associative_artifacts

    artifacts_dir = (root / args.artifacts_dir).resolve()
    output_dir = (
        (root / args.output_dir).resolve()
        if args.output_dir
        else root / "research" / "associative_memory_fft" / "deployment" / f"{args.task}_seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = checkpoint_path(artifacts_dir, args.task, args.seed)
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {ckpt}")

    with open(ckpt, "rb") as f:
        params = pickle.load(f)["params"]

    memory_state, checkpoints, _cfg = load_associative_artifacts(artifacts_dir, args.task, args.seed)
    bundle = load_clinical_bundle(
        ClinicalDatasetConfig(task=args.task, data_dir=(root / args.data_dir).resolve())
    )
    num_classes = int(bundle.num_classes)
    include_history = args.include_history and not args.no_history

    split_data = {
        "train": (bundle.x_train, bundle.y_train, bundle.sample_ids["train"]),
        "cal": (bundle.x_cal, bundle.y_cal, bundle.sample_ids["cal"]),
        "test": (bundle.x_test, bundle.y_test, bundle.sample_ids["test"]),
        "external": (bundle.x_external, bundle.y_external, bundle.sample_ids["external"]),
    }

    # Sanity on a small batch from cal
    x_cal = split_data["cal"][0]
    n_sanity = min(4, len(x_cal))
    sanity = run_deployment_sanity_checks(
        params,
        x_cal[:n_sanity],
        memory_state,
        num_classes=num_classes,
        checkpoints=checkpoints,
    )
    sanity_path = output_dir / "sanity_report.json"
    with open(sanity_path, "w", encoding="utf-8") as f:
        json.dump(
            {"passed": sanity.passed, "checks": sanity.checks, "details": sanity.details},
            f,
            indent=2,
        )
    if not sanity.passed:
        print("WARNING: deployment sanity checks failed:", sanity.checks)

    eval_frames: dict[str, object] = {}
    history_for_plots: list = []

    for split in args.splits:
        x, y, ids = split_data[split]
        if args.max_samples is not None:
            x, y, ids = x[: args.max_samples], y[: args.max_samples], ids[: args.max_samples]
        records, df = run_deployment_on_split(
            params,
            x,
            y,
            ids,
            memory_state,
            checkpoints,
            num_classes=num_classes,
            include_history=include_history,
        )
        if include_history and not history_for_plots:
            history_for_plots = records[: min(8, len(records))]
        paths = save_deployment_records(records, output_dir / split, prefix=f"{split}_deployment")
        summary = summarize_deployment_diagnostics(df)
        summary.to_csv(output_dir / split / f"{split}_diagnostic_summary.csv", index=False)
        eval_frames[split] = df
        print(f"{split}: wrote {paths['csv']} ({len(records)} samples)")

    if "cal" in eval_frames and "external" in eval_frames:
        cal_df = eval_frames["cal"]
        ext_df = eval_frames["external"]
        offline = offline_evaluate_memory_information(cal_df, ext_df)
        offline_path = output_dir / "offline_evaluation.csv"
        offline.to_csv(offline_path, index=False)
        print(f"offline evaluation: {offline_path}")

    if "external" in eval_frames:
        plot_deployment_diagnostics(
            eval_frames["external"],
            output_dir / "plots",
            prefix="external",
            history_records=history_for_plots if include_history else None,
        )
        print(f"plots: {output_dir / 'plots'}")

    print(f"done → {output_dir}")
    return 0 if sanity.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
