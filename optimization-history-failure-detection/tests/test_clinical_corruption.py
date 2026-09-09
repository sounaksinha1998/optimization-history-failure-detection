"""Tests for clinical image corruption utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.common.clinical_corruption import (
    CORRUPTION_NAMES,
    DEFAULT_CORRUPTION_SUITE,
    CorruptionSpec,
    apply_corruption,
    build_corrupted_set,
)


@pytest.mark.parametrize("channels", [1, 3])
def test_jpeg_roundtrip_preserves_shape(channels: int) -> None:
    x = np.random.default_rng(0).random((4, 28, 28, channels), dtype=np.float32)
    out = apply_corruption(x, CorruptionSpec("jpeg_compression", 3), seed=0)
    assert out.shape == x.shape
    assert out.dtype == np.float32
    assert np.all((out >= 0.0) & (out <= 1.0))


@pytest.mark.parametrize("channels", [1, 3])
def test_all_corruptions_accept_medmnist_shapes(channels: int) -> None:
    x = np.random.default_rng(1).random((8, 28, 28, channels), dtype=np.float32)
    for spec in DEFAULT_CORRUPTION_SUITE:
        out = apply_corruption(x, spec, seed=42)
        assert out.shape == x.shape, spec.name
        assert out.dtype == np.float32


def test_build_corrupted_set_keys() -> None:
    x = np.ones((2, 28, 28, 1), dtype=np.float32) * 0.5
    corrupted = build_corrupted_set(x, base_seed=7)
    assert set(corrupted) == {spec.key() for spec in DEFAULT_CORRUPTION_SUITE}
    assert set(CORRUPTION_NAMES) == {
        "gaussian_noise",
        "brightness",
        "contrast",
        "motion_blur",
        "jpeg_compression",
    }
