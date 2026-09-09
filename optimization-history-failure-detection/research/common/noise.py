"""Gaussian noise utilities for controlled MNIST corruption."""

from __future__ import annotations

import hashlib
from typing import Literal

import numpy as np

NoiseMode = Literal["static", "dynamic"]


def sample_noise_seed(sample_id: str, master_seed: int, *, epoch: int = 0, step: int = 0) -> int:
    """Derive a deterministic 32-bit seed from sample_id and optional step context."""
    key = f"{sample_id}:{master_seed}:{epoch}:{step}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def apply_gaussian_noise(
    x: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply x_noisy = clip(x + epsilon, 0, 1), epsilon ~ N(0, sigma^2)."""
    if sigma <= 0.0:
        return np.asarray(x, dtype=np.float32).copy()
    epsilon = rng.normal(loc=0.0, scale=sigma, size=x.shape).astype(np.float32)
    return np.clip(x.astype(np.float32) + epsilon, 0.0, 1.0)


def build_static_noisy_split(
    x_clean: np.ndarray,
    sample_ids: np.ndarray,
    sigma: float,
    master_seed: int,
) -> np.ndarray:
    """Generate fixed per-sample noise for an entire split."""
    x_clean = np.asarray(x_clean, dtype=np.float32)
    if sigma <= 0.0:
        return x_clean.copy()

    noisy = np.empty_like(x_clean)
    for i, sample_id in enumerate(sample_ids):
        seed = sample_noise_seed(str(sample_id), master_seed)
        rng = np.random.default_rng(seed)
        noisy[i] = apply_gaussian_noise(x_clean[i], sigma, rng)
    return noisy


def apply_dynamic_gaussian_noise(
    x: np.ndarray,
    sample_id: str,
    sigma: float,
    master_seed: int,
    *,
    epoch: int,
    step: int,
) -> np.ndarray:
    """Generate a fresh noise realization for one access."""
    seed = sample_noise_seed(sample_id, master_seed, epoch=epoch, step=step)
    rng = np.random.default_rng(seed)
    return apply_gaussian_noise(x, sigma, rng)
