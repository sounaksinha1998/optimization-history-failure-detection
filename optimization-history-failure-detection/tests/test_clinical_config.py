"""Tests for FinalClinicalConfig normalization."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.common.clinical_failure import FinalClinicalConfig


def test_single_dataset_string_normalized_to_tuple() -> None:
    cfg = FinalClinicalConfig(datasets="organamnist").resolve_paths()
    assert cfg.datasets == ("organamnist",)


def test_single_seed_int_normalized_to_tuple() -> None:
    cfg = FinalClinicalConfig(seeds=42).resolve_paths()
    assert cfg.seeds == (42,)


def test_unknown_dataset_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        FinalClinicalConfig(datasets="o").resolve_paths()


def test_ready_for_display_requires_scored_caches_for_configured_datasets(tmp_path: Path) -> None:
    from research.common.clinical_experiment import summarize_clinical_artifacts

    out = tmp_path / "final_experiment"
    (out / "metrics").mkdir(parents=True)
    (out / "figures").mkdir(parents=True)
    (out / "statistics").mkdir(parents=True)
    (out / "metrics" / "aggregate_metrics.csv").write_text("Dataset\npathmnist\n", encoding="utf-8")
    (out / "statistics" / "bootstrap_results.csv").write_text("task\npathmnist\n", encoding="utf-8")
    (out / "metrics" / "confident_wrong.csv").write_text("task\npathmnist\n", encoding="utf-8")
    (out / "figures" / "architecture.png").write_bytes(b"x")

    cfg = FinalClinicalConfig(datasets="organamnist", output_dir=out).resolve_paths()
    status = summarize_clinical_artifacts(cfg)
    assert status["final_exports_exist"] is True
    assert status["scoring_complete"] is False
    assert status["ready_for_display"] is False
