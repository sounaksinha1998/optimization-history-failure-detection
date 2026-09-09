"""Deterministic medical-image corruption suite for corrupted-ID evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

CORRUPTION_NAMES = ("gaussian_noise", "brightness", "contrast", "motion_blur", "jpeg_compression")


@dataclass(frozen=True)
class CorruptionSpec:
    name: str
    severity: int

    def key(self) -> str:
        return f"{self.name}_s{self.severity}"


DEFAULT_CORRUPTION_SUITE: tuple[CorruptionSpec, ...] = (
    CorruptionSpec("gaussian_noise", 1),
    CorruptionSpec("gaussian_noise", 3),
    CorruptionSpec("brightness", 2),
    CorruptionSpec("contrast", 2),
    CorruptionSpec("motion_blur", 2),
    CorruptionSpec("jpeg_compression", 3),
)


def corruption_config_dict(suite: tuple[CorruptionSpec, ...] = DEFAULT_CORRUPTION_SUITE) -> dict[str, Any]:
    return {
        "suite": [asdict(spec) for spec in suite],
        "deterministic": True,
        "severity_scale": {1: "mild", 2: "moderate", 3: "severe"},
    }


def _array_to_pil(arr: np.ndarray):
    """Convert HxW or HxWxC uint8 array to a PIL image."""
    from PIL import Image

    arr = np.asarray(arr, dtype=np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    if arr.ndim == 3:
        channels = arr.shape[-1]
        if channels == 1:
            return Image.fromarray(arr[..., 0], mode="L")
        if channels == 3:
            return Image.fromarray(arr, mode="RGB")
        if channels == 4:
            return Image.fromarray(arr, mode="RGBA")
    raise ValueError(f"Unsupported image array shape {arr.shape}")


def _pil_to_array(img, channels: int) -> np.ndarray:
    """Restore PIL image to float32 HxWxC in [0, 1] with the original channel count."""
    restored = np.asarray(img, dtype=np.float32) / 255.0
    if channels == 1:
        if restored.ndim == 2:
            return restored[..., None]
        return restored[..., :1]
    if channels == 3:
        if restored.ndim == 2:
            gray = restored
            return np.stack([gray, gray, gray], axis=-1)
        return restored[..., :3]
    if channels == 4:
        if restored.ndim == 2:
            gray = restored
            rgba = np.stack([gray, gray, gray, np.ones_like(gray)], axis=-1)
            return rgba
        return restored[..., :4]
    raise ValueError(f"Unsupported channel count {channels}")


def _jpeg_roundtrip(x: np.ndarray, quality: int) -> np.ndarray:
    try:
        from io import BytesIO

        from PIL import Image
    except ImportError:
        # Fallback without Pillow: quantize in float space.
        levels = max(4, quality // 8)
        return np.round(x * levels) / levels

    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected batched images (N,H,W,C); got shape {x.shape}")

    out = np.empty_like(x)
    channels = int(x.shape[-1])
    q = int(np.clip(quality, 10, 95))
    for i in range(x.shape[0]):
        arr = (np.clip(x[i], 0.0, 1.0) * 255.0).astype(np.uint8)
        img = _array_to_pil(arr)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        out[i] = _pil_to_array(Image.open(buf), channels)
    return out


def apply_corruption(x: np.ndarray, spec: CorruptionSpec, *, seed: int) -> np.ndarray:
    """Apply one corruption to a batch (N,H,W,C) in [0,1]."""
    x = np.asarray(x, dtype=np.float32)
    rng = np.random.default_rng(seed)
    sev = spec.severity

    if spec.name == "gaussian_noise":
        sigma = {1: 0.04, 2: 0.08, 3: 0.12}[sev]
        noise = rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
        return np.clip(x + noise, 0.0, 1.0)

    if spec.name == "brightness":
        delta = {1: 0.08, 2: 0.15, 3: 0.22}[sev]
        sign = 1.0 if sev % 2 == 0 else -1.0
        return np.clip(x + sign * delta, 0.0, 1.0)

    if spec.name == "contrast":
        factor = {1: 1.15, 2: 1.30, 3: 1.45}[sev]
        mean = x.mean(axis=(1, 2, 3), keepdims=True)
        return np.clip((x - mean) * factor + mean, 0.0, 1.0)

    if spec.name == "motion_blur":
        sigma = {1: 0.6, 2: 1.0, 3: 1.4}[sev]
        out = np.empty_like(x)
        for i in range(x.shape[0]):
            out[i] = gaussian_filter(x[i], sigma=(sigma, sigma, 0.0))
        return np.clip(out, 0.0, 1.0)

    if spec.name == "jpeg_compression":
        quality = {1: 70, 2: 45, 3: 25}[sev]
        return _jpeg_roundtrip(x, quality)

    raise ValueError(f"Unknown corruption: {spec.name}")


def build_corrupted_set(
    x: np.ndarray,
    *,
    suite: tuple[CorruptionSpec, ...] = DEFAULT_CORRUPTION_SUITE,
    base_seed: int = 0,
) -> dict[str, np.ndarray]:
    """Return one corrupted array per spec key."""
    out: dict[str, np.ndarray] = {}
    for i, spec in enumerate(suite):
        out[spec.key()] = apply_corruption(x, spec, seed=base_seed + i * 17)
    return out
