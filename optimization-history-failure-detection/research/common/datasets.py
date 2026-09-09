"""Controlled noisy MNIST dataset loading and caching."""

from __future__ import annotations

import gzip
import json
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from research.common.noise import apply_dynamic_gaussian_noise, build_static_noisy_split

NOISY_MNIST_VARIANTS: dict[str, float] = {
    "mnist_clean": 0.0,
    "mnist_noise_01": 0.1,
    "mnist_noise_025": 0.25,
    "mnist_noise_05": 0.5,
}

NoiseMode = Literal["static", "dynamic"]

_MNIST_IDX_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

_MNIST_DOWNLOAD_URLS = {
    "train-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}


@dataclass(frozen=True)
class NoisyMNISTConfig:
    variant: str
    noise_mode: NoiseMode = "static"
    split_seed: int = 42
    val_frac: float = 0.1
    train_subset: int | None = 10_000
    subset_indices_seed: int = 42
    noise_master_seed: int = 123
    data_dir: Path = Path("data")


@dataclass(frozen=True)
class NoisyMNISTBundle:
    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    sample_ids: dict[str, np.ndarray]
    metadata: pd.DataFrame
    config: NoisyMNISTConfig
    sigma: float
    mnist_indices: dict[str, np.ndarray]

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (28, 28, 1)

    @property
    def num_outputs(self) -> int:
        return 10


def list_noisy_mnist_variants() -> list[str]:
    return list(NOISY_MNIST_VARIANTS.keys())


def variant_sigma(variant: str) -> float:
    if variant not in NOISY_MNIST_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}. Choose from {list(NOISY_MNIST_VARIANTS)}")
    return NOISY_MNIST_VARIANTS[variant]


def noisy_mnist_root(data_dir: Path) -> Path:
    return Path(data_dir) / "noisy_mnist"


def manifest_path(data_dir: Path) -> Path:
    return noisy_mnist_root(data_dir) / "manifest.json"


def load_manifest(data_dir: Path) -> dict[str, Any]:
    path = manifest_path(data_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing manifest at {path}. Run scripts/build_noisy_mnist.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _mnist_raw_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "mnist" / "MNIST" / "raw"


def _ensure_mnist_raw_files(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for fname, url in _MNIST_DOWNLOAD_URLS.items():
        dest = raw_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            continue
        print(f"Downloading {fname} ...")
        urllib.request.urlretrieve(url, dest)


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


def load_mnist_numpy(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load raw MNIST train/test arrays in [0, 1] float32."""
    raw_dir = _mnist_raw_dir(data_dir)
    _ensure_mnist_raw_files(raw_dir)
    x_train = _read_idx_images(raw_dir / _MNIST_IDX_FILES["train_images"])
    y_train = _read_idx_labels(raw_dir / _MNIST_IDX_FILES["train_labels"])
    x_test = _read_idx_images(raw_dir / _MNIST_IDX_FILES["test_images"])
    y_test = _read_idx_labels(raw_dir / _MNIST_IDX_FILES["test_labels"])
    return x_train, y_train, x_test, y_test


def split_mnist(
    x_train_full: np.ndarray,
    y_train_full: np.ndarray,
    *,
    split_seed: int,
    val_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train/val split from official MNIST train set; test set unchanged."""
    indices = np.arange(len(x_train_full))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_frac,
        random_state=split_seed,
        stratify=y_train_full,
    )
    return (
        x_train_full[train_idx],
        y_train_full[train_idx],
        train_idx.astype(np.int32),
        x_train_full[val_idx],
        y_train_full[val_idx],
        val_idx.astype(np.int32),
    )


def make_sample_ids(split: str, count: int) -> np.ndarray:
    return np.array([f"{split}_{i:05d}" for i in range(count)], dtype=object)


def select_train_subset(
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_ids: np.ndarray,
    train_mnist_idx: np.ndarray,
    train_subset: int | None,
    subset_indices_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if train_subset is None or train_subset >= len(x_train):
        return x_train, y_train, train_ids, train_mnist_idx
    rng = np.random.default_rng(subset_indices_seed)
    chosen = np.sort(rng.choice(len(x_train), size=train_subset, replace=False))
    return (
        x_train[chosen],
        y_train[chosen],
        train_ids[chosen],
        train_mnist_idx[chosen],
    )


def build_sample_index_rows(
    splits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name, (y_split, sample_ids, mnist_idx) in splits.items():
        for label, sample_id, mnist_index in zip(y_split, sample_ids, mnist_idx):
            rows.append(
                {
                    "sample_id": str(sample_id),
                    "split": split_name,
                    "label": int(label),
                    "mnist_index": int(mnist_index),
                }
            )
    return pd.DataFrame(rows)


def build_metadata_table(
    sample_index: pd.DataFrame,
    *,
    noise_level: float,
    noise_mode: NoiseMode,
    noise_seed: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": sample_index["sample_id"],
            "label": sample_index["label"],
            "noise_level": noise_level,
            "noise_mode": noise_mode,
            "noise_seed": noise_seed,
        }
    )


def save_variant_cache(
    variant_dir: Path,
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    metadata: pd.DataFrame,
) -> None:
    variant_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        variant_dir / "arrays.npz",
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )
    metadata.to_csv(variant_dir / "metadata.csv", index=False)


def load_variant_cache(variant_dir: Path) -> dict[str, np.ndarray]:
    with np.load(variant_dir / "arrays.npz") as data:
        return {key: data[key] for key in data.files}


def build_noisy_mnist_dataset(
    *,
    data_dir: Path,
    split_seed: int = 42,
    val_frac: float = 0.1,
    train_subset: int | None = 10_000,
    subset_indices_seed: int = 42,
    noise_master_seed: int = 123,
    variants: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Download MNIST, split, assign sample_ids, and materialize all static variants."""
    variants = variants or NOISY_MNIST_VARIANTS
    root = noisy_mnist_root(data_dir)
    root.mkdir(parents=True, exist_ok=True)

    x_train_full, y_train_full, x_test, y_test = load_mnist_numpy(data_dir)
    x_train, y_train, train_mnist_idx, x_val, y_val, val_mnist_idx = split_mnist(
        x_train_full,
        y_train_full,
        split_seed=split_seed,
        val_frac=val_frac,
    )
    test_mnist_idx = np.arange(len(x_test), dtype=np.int32)

    train_ids = make_sample_ids("train", len(x_train))
    val_ids = make_sample_ids("val", len(x_val))
    test_ids = make_sample_ids("test", len(x_test))

    x_train, y_train, train_ids, train_mnist_idx = select_train_subset(
        x_train,
        y_train,
        train_ids,
        train_mnist_idx,
        train_subset,
        subset_indices_seed,
    )

    splits_clean = {
        "train": (y_train, train_ids, train_mnist_idx),
        "val": (y_val, val_ids, val_mnist_idx),
        "test": (y_test, test_ids, test_mnist_idx),
    }
    sample_index = build_sample_index_rows(splits_clean)
    sample_index.to_csv(root / "sample_index.csv", index=False)

    clean_splits = {
        "train": x_train,
        "val": x_val,
        "test": x_test,
    }

    manifest = {
        "split_seed": split_seed,
        "val_frac": val_frac,
        "train_subset": train_subset,
        "subset_indices_seed": subset_indices_seed,
        "noise_master_seed": noise_master_seed,
        "variants": {name: {"sigma": sigma} for name, sigma in variants.items()},
    }
    manifest_path(data_dir).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for variant_name, sigma in variants.items():
        variant_dir = root / variant_name
        x_train_n = build_static_noisy_split(x_train, train_ids, sigma, noise_master_seed)
        x_val_n = build_static_noisy_split(x_val, val_ids, sigma, noise_master_seed)
        x_test_n = build_static_noisy_split(x_test, test_ids, sigma, noise_master_seed)
        metadata = build_metadata_table(
            sample_index,
            noise_level=sigma,
            noise_mode="static",
            noise_seed=noise_master_seed,
        )
        save_variant_cache(
            variant_dir,
            x_train=x_train_n,
            y_train=y_train,
            x_val=x_val_n,
            y_val=y_val,
            x_test=x_test_n,
            y_test=y_test,
            metadata=metadata,
        )

    return manifest


class NoisyMNISTLoader:
    """Optional dynamic-noise view over a clean split."""

    def __init__(
        self,
        x_clean: np.ndarray,
        sample_ids: np.ndarray,
        *,
        sigma: float,
        noise_master_seed: int,
        noise_mode: NoiseMode = "static",
        x_static: np.ndarray | None = None,
    ) -> None:
        self.x_clean = x_clean
        self.sample_ids = sample_ids
        self.sigma = sigma
        self.noise_master_seed = noise_master_seed
        self.noise_mode = noise_mode
        self.x_static = x_static

    def __len__(self) -> int:
        return len(self.x_clean)

    def __getitem__(self, index: int, *, epoch: int = 0, step: int = 0) -> np.ndarray:
        if self.noise_mode == "static":
            if self.x_static is None:
                return self.x_clean[index]
            return self.x_static[index]
        return apply_dynamic_gaussian_noise(
            self.x_clean[index],
            str(self.sample_ids[index]),
            self.sigma,
            self.noise_master_seed,
            epoch=epoch,
            step=step,
        )


def load_noisy_mnist(config: NoisyMNISTConfig) -> NoisyMNISTBundle:
    """Load a noisy MNIST variant from cache or apply dynamic noise on the fly."""
    sigma = variant_sigma(config.variant)
    manifest = load_manifest(config.data_dir)
    root = noisy_mnist_root(config.data_dir)
    sample_index = pd.read_csv(root / "sample_index.csv")
    variant_dir = root / config.variant

    if config.noise_mode == "static":
        if not variant_dir.exists():
            raise FileNotFoundError(
                f"Missing variant cache at {variant_dir}. Run scripts/build_noisy_mnist.py."
            )
        arrays = load_variant_cache(variant_dir)
        metadata = pd.read_csv(variant_dir / "metadata.csv")
        x_train, y_train = arrays["x_train"], arrays["y_train"]
        x_val, y_val = arrays["x_val"], arrays["y_val"]
        x_test, y_test = arrays["x_test"], arrays["y_test"]
    else:
        clean_arrays = load_variant_cache(root / "mnist_clean")
        x_train, y_train = clean_arrays["x_train"], clean_arrays["y_train"]
        x_val, y_val = clean_arrays["x_val"], clean_arrays["y_val"]
        x_test, y_test = clean_arrays["x_test"], clean_arrays["y_test"]
        metadata = build_metadata_table(
            sample_index,
            noise_level=sigma,
            noise_mode="dynamic",
            noise_seed=config.noise_master_seed,
        )

    train_ids = sample_index.loc[sample_index["split"] == "train", "sample_id"].to_numpy()
    val_ids = sample_index.loc[sample_index["split"] == "val", "sample_id"].to_numpy()
    test_ids = sample_index.loc[sample_index["split"] == "test", "sample_id"].to_numpy()

    train_mnist_idx = sample_index.loc[sample_index["split"] == "train", "mnist_index"].to_numpy(dtype=np.int32)
    val_mnist_idx = sample_index.loc[sample_index["split"] == "val", "mnist_index"].to_numpy(dtype=np.int32)
    test_mnist_idx = sample_index.loc[sample_index["split"] == "test", "mnist_index"].to_numpy(dtype=np.int32)

    if config.noise_mode == "dynamic":
        if sigma > 0.0 and config.variant != "mnist_clean":
            pass  # arrays stay clean; apply noise per batch in the training loop
    return NoisyMNISTBundle(
        name=config.variant,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        sample_ids={"train": train_ids, "val": val_ids, "test": test_ids},
        metadata=metadata,
        config=config,
        sigma=sigma,
        mnist_indices={
            "train": train_mnist_idx,
            "val": val_mnist_idx,
            "test": test_mnist_idx,
        },
    )
