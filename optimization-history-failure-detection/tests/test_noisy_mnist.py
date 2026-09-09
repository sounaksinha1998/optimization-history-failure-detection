"""Tests for controlled noisy MNIST dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.common.datasets import (
    NOISY_MNIST_VARIANTS,
    NoisyMNISTConfig,
    build_noisy_mnist_dataset,
    build_static_noisy_split,
    load_noisy_mnist,
    make_sample_ids,
    split_mnist,
)
from research.common.noise import apply_dynamic_gaussian_noise, sample_noise_seed


@pytest.fixture(scope="module")
def built_dataset(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("noisy_mnist_data")
    build_noisy_mnist_dataset(
        data_dir=data_dir,
        split_seed=42,
        val_frac=0.1,
        train_subset=512,
        subset_indices_seed=42,
        noise_master_seed=123,
        variants=NOISY_MNIST_VARIANTS,
    )
    return data_dir


def test_split_stability():
    rng_x = np.random.default_rng(0)
    x = rng_x.random((200, 28, 28, 1), dtype=np.float32)
    y = rng_x.integers(0, 10, size=200)
    split_a = split_mnist(x, y, split_seed=42, val_frac=0.1)
    split_b = split_mnist(x, y, split_seed=42, val_frac=0.1)
    assert np.array_equal(split_a[2], split_b[2])
    ids_a = make_sample_ids("train", len(split_a[0]))
    ids_b = make_sample_ids("train", len(split_b[0]))
    assert np.array_equal(ids_a, ids_b)


def test_cross_variant_sample_id_alignment(built_dataset):
    bundles = [
        load_noisy_mnist(NoisyMNISTConfig(variant=v, data_dir=built_dataset))
        for v in NOISY_MNIST_VARIANTS
    ]
    ref = bundles[0].sample_ids
    for bundle in bundles[1:]:
        assert np.array_equal(ref["train"], bundle.sample_ids["train"])
        assert np.array_equal(ref["val"], bundle.sample_ids["val"])
        assert np.array_equal(ref["test"], bundle.sample_ids["test"])


def test_static_reproducibility():
    x = np.random.default_rng(1).random((4, 28, 28, 1), dtype=np.float32)
    ids = make_sample_ids("train", 4)
    a = build_static_noisy_split(x, ids, sigma=0.25, master_seed=123)
    b = build_static_noisy_split(x, ids, sigma=0.25, master_seed=123)
    assert np.allclose(a, b)


def test_sigma_ordering(built_dataset):
    clean = load_noisy_mnist(NoisyMNISTConfig(variant="mnist_clean", data_dir=built_dataset))
    noisy_variants = ["mnist_noise_01", "mnist_noise_025", "mnist_noise_05"]
    mean_norms = []
    for variant in noisy_variants:
        bundle = load_noisy_mnist(NoisyMNISTConfig(variant=variant, data_dir=built_dataset))
        diff = bundle.x_train - clean.x_train
        mean_norms.append(float(np.mean(np.linalg.norm(diff.reshape(len(diff), -1), axis=1))))
    assert mean_norms == sorted(mean_norms)


def test_labels_preserved(built_dataset):
    for variant in NOISY_MNIST_VARIANTS:
        bundle = load_noisy_mnist(NoisyMNISTConfig(variant=variant, data_dir=built_dataset))
        clean = load_noisy_mnist(NoisyMNISTConfig(variant="mnist_clean", data_dir=built_dataset))
        assert np.array_equal(bundle.y_train, clean.y_train)
        assert np.array_equal(bundle.y_val, clean.y_val)
        assert np.array_equal(bundle.y_test, clean.y_test)


def test_manifest_and_metadata(built_dataset):
    manifest = json.loads((built_dataset / "noisy_mnist" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["split_seed"] == 42
    assert manifest["noise_master_seed"] == 123
    assert set(manifest["variants"]) == set(NOISY_MNIST_VARIANTS)

    sample_index = pd.read_csv(built_dataset / "noisy_mnist" / "sample_index.csv")
    metadata = pd.read_csv(built_dataset / "noisy_mnist" / "mnist_noise_025" / "metadata.csv")
    assert len(sample_index) == len(metadata)
    assert set(sample_index["sample_id"]) == set(metadata["sample_id"])
    assert (metadata["noise_level"] == 0.25).all()
    assert (metadata["noise_mode"] == "static").all()


def test_dynamic_differs_static():
    x = np.random.default_rng(2).random((28, 28, 1), dtype=np.float32)
    sample_id = "train_00000"
    a = apply_dynamic_gaussian_noise(x, sample_id, 0.25, 123, epoch=0, step=0)
    b = apply_dynamic_gaussian_noise(x, sample_id, 0.25, 123, epoch=0, step=1)
    assert not np.allclose(a, b)


def test_dynamic_seed_deterministic():
    x = np.random.default_rng(3).random((28, 28, 1), dtype=np.float32)
    sample_id = "train_00001"
    a = apply_dynamic_gaussian_noise(x, sample_id, 0.25, 123, epoch=1, step=5)
    b = apply_dynamic_gaussian_noise(x, sample_id, 0.25, 123, epoch=1, step=5)
    assert np.allclose(a, b)


def test_sample_noise_seed_stable():
    assert sample_noise_seed("train_00000", 123) == sample_noise_seed("train_00000", 123)
    assert sample_noise_seed("train_00000", 123) != sample_noise_seed("train_00001", 123)
