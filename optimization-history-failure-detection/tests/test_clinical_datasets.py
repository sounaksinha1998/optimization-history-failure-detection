"""Tests for clinical dataset loaders."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.common.clinical_datasets import (
    ClinicalDatasetConfig,
    build_dataset_manifest,
    load_clinical_bundle,
)


def test_pathmnist_bundle_shapes(tmp_path: Path) -> None:
    cfg = ClinicalDatasetConfig(
        task="pathmnist",
        data_dir=tmp_path / "clinical",
        max_train=32,
        max_cal=16,
        max_test=16,
        max_external=16,
    )
    bundle = load_clinical_bundle(cfg)
    assert bundle.x_train.shape[-1] in (1, 3)
    assert len(bundle.x_train) == 32
    assert len(bundle.x_cal) == 16
    assert len(bundle.x_test) == 16
    assert len(bundle.x_external) == 16
    assert bundle.num_classes == 9


def test_dataset_manifest(tmp_path: Path) -> None:
    cfg = ClinicalDatasetConfig(task="dermamnist", data_dir=tmp_path / "clinical", max_train=8, max_cal=4, max_test=4, max_external=4)
    bundle = load_clinical_bundle(cfg)
    manifest = build_dataset_manifest({"dermamnist": bundle}, data_dir=cfg.data_dir)
    assert "dermamnist" in manifest["datasets"]
    assert manifest["datasets"]["dermamnist"]["num_classes"] == 7


def test_calibration_disjoint_from_test(tmp_path: Path) -> None:
    cfg = ClinicalDatasetConfig(task="organamnist", data_dir=tmp_path / "clinical", max_train=16, max_cal=8, max_test=8, max_external=8)
    bundle = load_clinical_bundle(cfg)
    cal_ids = set(bundle.sample_ids["cal"])
    train_ids = set(bundle.sample_ids["train"])
    assert cal_ids.isdisjoint(train_ids)
