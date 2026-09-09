"""Controlled distribution-shift datasets for Phase 5 OOD evaluation."""

from __future__ import annotations

import gzip
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from research.common.datasets import load_mnist_numpy, make_sample_ids
from research.common.noise import apply_gaussian_noise, sample_noise_seed

DEFAULT_OOD_SHIFTS: dict[str, dict[str, Any]] = {
    "rotated": {"angle_deg": 45.0},
    "translated": {"dx": 4, "dy": 4},
    "corrupted": {"sigma": 0.25},
    "fashion_mnist": {},
}

_FASHION_MNIST_FILES = {
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

_FASHION_MNIST_URLS = {
    "t10k-images-idx3-ubyte.gz": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
}


@dataclass(frozen=True)
class ShiftDataset:
    name: str
    x: np.ndarray
    y: np.ndarray
    sample_ids: np.ndarray


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad image magic {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols, 1).astype(np.float32) / 255.0


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Bad label magic {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int32)


def _bilinear_sample(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = img.shape
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    return wa * img[y0, x0] + wb * img[y1, x0] + wc * img[y0, x1] + wd * img[y1, x1]


def rotate_images(x: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate each image counter-clockwise by angle_deg."""
    x = np.asarray(x, dtype=np.float32)
    if angle_deg == 0.0:
        return x.copy()
    angle = np.deg2rad(angle_deg)
    cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
    out = np.empty_like(x)
    for i in range(len(x)):
        img = x[i, :, :, 0]
        h, w = img.shape
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        x_rel = xs - cx
        y_rel = ys - cy
        src_x = cos_a * x_rel + sin_a * y_rel + cx
        src_y = -sin_a * x_rel + cos_a * y_rel + cy
        valid = (src_x >= 0) & (src_x <= w - 1) & (src_y >= 0) & (src_y <= h - 1)
        sampled = np.zeros((h, w), dtype=np.float32)
        sampled[valid] = _bilinear_sample(img, src_x[valid], src_y[valid])
        out[i, :, :, 0] = sampled
    return out


def translate_images(x: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate images by (dx, dy) with zero padding."""
    x = np.asarray(x, dtype=np.float32)
    if dx == 0 and dy == 0:
        return x.copy()
    out = np.zeros_like(x)
    h, w = x.shape[1], x.shape[2]
    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx)
    dst_y0 = max(0, dy)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x0 = max(0, dx)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    out[:, dst_y0:dst_y1, dst_x0:dst_x1, :] = x[:, src_y0:src_y1, src_x0:src_x1, :]
    return out


def corrupt_images(
    x: np.ndarray,
    sample_ids: np.ndarray,
    *,
    sigma: float,
    master_seed: int = 123,
) -> np.ndarray:
    """Apply deterministic Gaussian corruption per sample."""
    x = np.asarray(x, dtype=np.float32)
    if sigma <= 0.0:
        return x.copy()
    out = np.empty_like(x)
    for i, sample_id in enumerate(sample_ids):
        seed = sample_noise_seed(str(sample_id), master_seed)
        rng = np.random.default_rng(seed)
        out[i] = apply_gaussian_noise(x[i], sigma, rng)
    return out


def _fashion_mnist_raw_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "fashion_mnist" / "raw"


def _ensure_fashion_mnist_raw_files(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for fname, url in _FASHION_MNIST_URLS.items():
        dest = raw_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            continue
        print(f"Downloading {fname} ...")
        urllib.request.urlretrieve(url, dest)


def load_fashion_mnist_test(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    raw_dir = _fashion_mnist_raw_dir(data_dir)
    _ensure_fashion_mnist_raw_files(raw_dir)
    x_test = _read_idx_images(raw_dir / _FASHION_MNIST_FILES["test_images"])
    y_test = _read_idx_labels(raw_dir / _FASHION_MNIST_FILES["test_labels"])
    return x_test, y_test


def build_shift_dataset(
    *,
    shift_name: str,
    x_source: np.ndarray,
    y_source: np.ndarray,
    sample_ids: np.ndarray,
    shift_params: dict[str, Any] | None = None,
    data_dir: Path | None = None,
) -> ShiftDataset:
    """Materialize one OOD shift dataset from a source split."""
    params = dict(shift_params or DEFAULT_OOD_SHIFTS.get(shift_name, {}))

    if shift_name == "rotated":
        x_shift = rotate_images(x_source, float(params.get("angle_deg", 45.0)))
        y_shift = y_source
        ids = sample_ids
    elif shift_name == "translated":
        x_shift = translate_images(x_source, int(params.get("dx", 4)), int(params.get("dy", 4)))
        y_shift = y_source
        ids = sample_ids
    elif shift_name == "corrupted":
        x_shift = corrupt_images(
            x_source,
            sample_ids,
            sigma=float(params.get("sigma", 0.25)),
            master_seed=int(params.get("master_seed", 123)),
        )
        y_shift = y_source
        ids = sample_ids
    elif shift_name == "fashion_mnist":
        if data_dir is None:
            raise ValueError("data_dir is required for fashion_mnist shift")
        x_shift, y_shift = load_fashion_mnist_test(Path(data_dir))
        ids = make_sample_ids("fashion_test", len(x_shift))
    else:
        raise ValueError(f"Unknown shift {shift_name!r}. Choose from {list(DEFAULT_OOD_SHIFTS)}")

    return ShiftDataset(name=shift_name, x=x_shift, y=y_shift, sample_ids=ids)


def load_id_test_split(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return MNIST official test split with stable sample IDs."""
    _x_train, _y_train, x_test, y_test = load_mnist_numpy(Path(data_dir))
    sample_ids = make_sample_ids("test", len(x_test))
    return x_test, y_test, sample_ids


def list_default_ood_shifts() -> list[str]:
    return list(DEFAULT_OOD_SHIFTS.keys())
