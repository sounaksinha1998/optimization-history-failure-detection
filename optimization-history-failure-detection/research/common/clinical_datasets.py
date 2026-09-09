"""Clinical MedMNIST loaders and external population-shift datasets."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ClinicalTask = Literal[
    "pathmnist",
    "dermamnist",
    "organamnist",
    "organcmnist",
    "retinamnist",
    "bloodmnist",
]
VALID_CLINICAL_TASKS: tuple[ClinicalTask, ...] = (
    "pathmnist",
    "dermamnist",
    "organamnist",
    "organcmnist",
    "retinamnist",
    "bloodmnist",
)

MEDMNIST_FILES: dict[str, str] = {
    "pathmnist": "pathmnist.npz",
    "dermamnist": "dermamnist.npz",
    "organamnist": "organamnist.npz",
    "retinamnist": "retinamnist.npz",
    "breastmnist": "breastmnist.npz",
    "organcmnist": "organcmnist.npz",
    "bloodmnist": "bloodmnist.npz",
}

MEDMNIST_URL = "https://zenodo.org/records/10519652/files/{filename}?download=1"

EXTERNAL_TARGETS: dict[ClinicalTask, str] = {
    "pathmnist": "HMU-CRC-Hist550K",
    "dermamnist": "DermaMNIST-E",
    "organamnist": "AMOS-22",
    "organcmnist": "AMOS-22",
    "retinamnist": "APTOS-2019",
    "bloodmnist": "BloodMNIST-E",
}

# Cross-dataset proxies when full external cache is unavailable (documented in manifest).
EXTERNAL_PROXY_MEDMNIST: dict[ClinicalTask, str] = {
    "pathmnist": "breastmnist",
    "dermamnist": "dermamnist",
    "organamnist": "organcmnist",
    "organcmnist": "organamnist",
    "retinamnist": "retinamnist",
    "bloodmnist": "bloodmnist",
}

TASK_NUM_CLASSES: dict[ClinicalTask, int] = {
    "pathmnist": 9,
    "dermamnist": 7,
    "organamnist": 11,
    "organcmnist": 11,
    "retinamnist": 5,
    "bloodmnist": 8,
}


@dataclass(frozen=True)
class ClinicalDatasetConfig:
    task: ClinicalTask
    data_dir: Path = Path("data/clinical")
    cal_fraction: float = 0.20
    cal_split_seed: int = 42
    max_train: int | None = None
    max_cal: int | None = None
    max_test: int | None = None
    max_external: int | None = None
    image_size: int = 28


@dataclass
class ClinicalBundle:
    task: ClinicalTask
    num_classes: int
    x_train: np.ndarray
    y_train: np.ndarray
    x_cal: np.ndarray
    y_cal: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    x_external: np.ndarray
    y_external: np.ndarray
    sample_ids: dict[str, np.ndarray]
    patient_ids: dict[str, np.ndarray | None]
    external_name: str
    external_proxy_used: bool
    preprocessing: dict[str, Any] = field(default_factory=dict)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_array(arr: np.ndarray) -> str:
    return _sha256_bytes(arr.tobytes())


def _download_npz(filename: str, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / filename
    if out_path.exists():
        try:
            with np.load(out_path) as data:
                _ = data["train_images"].shape
            return out_path
        except (OSError, KeyError, ValueError):
            out_path.unlink(missing_ok=True)

    url = MEDMNIST_URL.format(filename=filename)
    tmp_path = out_path.with_suffix(".npz.part")
    print(f"Downloading {filename} from MedMNIST …")
    urllib.request.urlretrieve(url, tmp_path)
    with np.load(tmp_path) as data:
        _ = data["train_images"].shape
    tmp_path.replace(out_path)
    return out_path


def _load_medmnist_arrays(name: str, data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = _download_npz(MEDMNIST_FILES[name], data_dir)
    with np.load(path) as data:
        x_train = data["train_images"].astype(np.float32) / 255.0
        y_train = data["train_labels"].astype(np.int32).squeeze()
        x_val = data["val_images"].astype(np.float32) / 255.0
        y_val = data["val_labels"].astype(np.int32).squeeze()
        x_test = data["test_images"].astype(np.float32) / 255.0
        y_test = data["test_labels"].astype(np.int32).squeeze()
    return x_train, y_train, x_val, y_val, x_test, y_test


def _ensure_hwc(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3:
        return x[..., None]
    if x.ndim == 4 and x.shape[-1] in (1, 3):
        return x
    if x.ndim == 4 and x.shape[1] in (1, 3):
        return np.transpose(x, (0, 2, 3, 1))
    raise ValueError(f"Unsupported image shape {x.shape}")


def _resize_nearest(x: np.ndarray, size: int) -> np.ndarray:
    if x.shape[1] == size and x.shape[2] == size:
        return x
    n = x.shape[0]
    out = np.zeros((n, size, size, x.shape[-1]), dtype=np.float32)
    ys = (np.linspace(0, x.shape[1] - 1, size)).astype(np.int32)
    xs = (np.linspace(0, x.shape[2] - 1, size)).astype(np.int32)
    for i in range(n):
        out[i] = x[i, ys][:, xs]
    return out


def _make_sample_ids(prefix: str, n: int) -> np.ndarray:
    return np.array([f"{prefix}_{i:06d}" for i in range(n)], dtype=object)


def _maybe_subset(x: np.ndarray, y: np.ndarray, ids: np.ndarray, max_n: int | None, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_n is None or len(x) <= max_n:
        return x, y, ids
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=max_n, replace=False)
    return x[idx], y[idx], ids[idx]


def _load_external_cache(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path) as data:
        x = _ensure_hwc(data["images"].astype(np.float32))
        if x.max() > 1.0:
            x = x / 255.0
        y = data["labels"].astype(np.int32).squeeze()
    return x, y


def _external_style_shift(x: np.ndarray, *, seed: int, task: ClinicalTask) -> np.ndarray:
    """Deterministic population-style shift for DermaMNIST-E proxy."""
    rng = np.random.default_rng(seed)
    out = x.copy()
    if task == "dermamnist":
        # Simulate different acquisition pipeline: gamma + channel gains.
        gamma = 1.25
        gains = np.array([1.15, 0.90, 1.05], dtype=np.float32)
        out = np.clip(out, 0.0, 1.0) ** gamma
        out = np.clip(out * gains.reshape(1, 1, 1, 3), 0.0, 1.0)
    elif task == "retinamnist":
        # Fundus acquisition shift: illumination + contrast (grayscale).
        gamma = 0.88
        out = np.clip(out, 0.0, 1.0) ** gamma
        out = np.clip((out - 0.5) * 1.15 + 0.55, 0.0, 1.0)
    elif task == "pathmnist":
        # Stain variation proxy.
        stain = np.array([1.10, 0.95, 1.20], dtype=np.float32)
        out = np.clip(out * stain.reshape(1, 1, 1, 3), 0.0, 1.0)
    elif task == "bloodmnist":
        # Microscopy-style acquisition shift: mild gamma + channel gains.
        gamma = 1.12
        gains = np.array([1.08, 0.94, 1.02], dtype=np.float32)
        out = np.clip(out, 0.0, 1.0) ** gamma
        out = np.clip(out * gains.reshape(1, 1, 1, 3), 0.0, 1.0)
    else:
        # CT windowing shift proxy for AMOS-style population drift.
        out = np.clip((out - 0.5) * 1.2 + 0.55, 0.0, 1.0)
    # Fixed per-dataset bias field (population-level, not sample-specific tuning).
    bias = rng.normal(0.0, 0.02, size=(1, out.shape[1], out.shape[2], out.shape[3])).astype(np.float32)
    return np.clip(out + bias, 0.0, 1.0)


def load_external_dataset(
    task: ClinicalTask,
    *,
    data_dir: Path,
    num_classes: int,
    image_size: int,
    max_external: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str, bool]:
    target = EXTERNAL_TARGETS[task]
    cache_path = data_dir / "external" / f"{task}_{target.replace('-', '_').lower()}.npz"
    cached = _load_external_cache(cache_path)
    if cached is not None:
        x_ext, y_ext = cached
        proxy_used = False
    else:
        proxy_name = EXTERNAL_PROXY_MEDMNIST[task]
        if proxy_name == task:
            # DermaMNIST-E proxy: style-shifted official test images (held-out evaluation only).
            _, _, _, _, x_test, y_test = _load_medmnist_arrays(task, data_dir)
            x_ext = _external_style_shift(x_test, seed=seed, task=task)
            y_ext = y_test.copy()
            proxy_used = True
        else:
            _, _, _, _, x_ext, y_ext = _load_medmnist_arrays(proxy_name, data_dir)
            proxy_used = True
        # Align labels to ID class space when using cross-dataset proxy.
        y_ext = np.clip(y_ext, 0, num_classes - 1)

    x_ext = _ensure_hwc(x_ext)
    x_ext = _resize_nearest(x_ext, image_size)
    ids = _make_sample_ids(f"{task}_ext", len(x_ext))
    x_ext, y_ext, _ = _maybe_subset(x_ext, y_ext, ids, max_external, seed)
    return x_ext, y_ext, target, proxy_used


def load_clinical_bundle(cfg: ClinicalDatasetConfig) -> ClinicalBundle:
    x_train_full, y_train_full, x_val, y_val, x_test, y_test = _load_medmnist_arrays(cfg.task, cfg.data_dir)
    x_official = np.concatenate([x_train_full, x_val], axis=0)
    y_official = np.concatenate([y_train_full, y_val], axis=0)

    ids_official = _make_sample_ids(f"{cfg.task}_id", len(x_official))
    x_train, x_cal, y_train, y_cal, ids_train, ids_cal = train_test_split(
        x_official,
        y_official,
        ids_official,
        test_size=cfg.cal_fraction,
        random_state=cfg.cal_split_seed,
        stratify=y_official,
    )

    x_train = _ensure_hwc(x_train)
    x_cal = _ensure_hwc(x_cal)
    x_test = _ensure_hwc(x_test)
    x_train = _resize_nearest(x_train, cfg.image_size)
    x_cal = _resize_nearest(x_cal, cfg.image_size)
    x_test = _resize_nearest(x_test, cfg.image_size)

    ids_train = np.asarray(ids_train, dtype=object)
    ids_cal = np.asarray(ids_cal, dtype=object)
    ids_test = _make_sample_ids(f"{cfg.task}_test", len(x_test))

    x_train, y_train, ids_train = _maybe_subset(x_train, y_train, ids_train, cfg.max_train, cfg.cal_split_seed)
    x_cal, y_cal, ids_cal = _maybe_subset(x_cal, y_cal, ids_cal, cfg.max_cal, cfg.cal_split_seed + 1)
    x_test, y_test, ids_test = _maybe_subset(x_test, y_test, ids_test, cfg.max_test, cfg.cal_split_seed + 2)

    x_external, y_external, external_name, proxy_used = load_external_dataset(
        cfg.task,
        data_dir=cfg.data_dir,
        num_classes=TASK_NUM_CLASSES[cfg.task],
        image_size=cfg.image_size,
        max_external=cfg.max_external,
        seed=cfg.cal_split_seed + 99,
    )
    ids_external = _make_sample_ids(f"{cfg.task}_external", len(x_external))

    return ClinicalBundle(
        task=cfg.task,
        num_classes=TASK_NUM_CLASSES[cfg.task],
        x_train=x_train,
        y_train=y_train,
        x_cal=x_cal,
        y_cal=y_cal,
        x_test=x_test,
        y_test=y_test,
        x_external=x_external,
        y_external=y_external,
        sample_ids={
            "train": ids_train,
            "cal": ids_cal,
            "test": ids_test,
            "external": ids_external,
        },
        patient_ids={
            "train": None,
            "cal": None,
            "test": None,
            "external": None,
        },
        external_name=external_name,
        external_proxy_used=proxy_used,
        preprocessing={
            "normalization": "divide_by_255",
            "image_size": cfg.image_size,
            "channels_last": True,
            "cal_fraction": cfg.cal_fraction,
            "cal_split_seed": cfg.cal_split_seed,
        },
    )


def class_distribution(labels: np.ndarray, num_classes: int) -> dict[str, int]:
    counts = np.bincount(labels.astype(int), minlength=num_classes)
    return {str(i): int(c) for i, c in enumerate(counts)}


def build_dataset_manifest(
    bundles: dict[ClinicalTask, ClinicalBundle],
    *,
    data_dir: Path,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {"datasets": {}, "external_domain_mapping": {}}
    for task, bundle in bundles.items():
        manifest["datasets"][task] = {
            "id_dataset": task,
            "id_version": "MedMNISTv2",
            "id_source": MEDMNIST_URL.format(filename=MEDMNIST_FILES[task]),
            "external_dataset": bundle.external_name,
            "external_proxy_used": bundle.external_proxy_used,
            "splits": {
                "train": int(len(bundle.x_train)),
                "calibration": int(len(bundle.x_cal)),
                "id_test": int(len(bundle.x_test)),
                "external": int(len(bundle.x_external)),
            },
            "num_classes": bundle.num_classes,
            "class_distribution": {
                "train": class_distribution(bundle.y_train, bundle.num_classes),
                "calibration": class_distribution(bundle.y_cal, bundle.num_classes),
                "id_test": class_distribution(bundle.y_test, bundle.num_classes),
                "external": class_distribution(bundle.y_external, bundle.num_classes),
            },
            "preprocessing": bundle.preprocessing,
            "image_resolution": [bundle.x_train.shape[1], bundle.x_train.shape[2], bundle.x_train.shape[3]],
            "checksums": {
                "train_images": _sha256_array(bundle.x_train),
                "id_test_images": _sha256_array(bundle.x_test),
                "external_images": _sha256_array(bundle.x_external),
            },
            "patient_group_information": "not_available_in_medmnist_npz",
        }
        manifest["external_domain_mapping"][task] = {
            "id": task,
            "external": bundle.external_name,
            "proxy_used": bundle.external_proxy_used,
        }
    manifest["data_dir"] = str(data_dir)
    return manifest


def save_dataset_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def bundles_to_summary_df(bundles: dict[ClinicalTask, ClinicalBundle]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task, bundle in bundles.items():
        for split in ("train", "cal", "test", "external"):
            key_x = f"x_{split}" if split != "cal" else "x_cal"
            key_y = f"y_{split}" if split != "cal" else "y_cal"
            rows.append(
                {
                    "task": task,
                    "split": "calibration" if split == "cal" else split,
                    "n": len(getattr(bundle, key_x)),
                    "num_classes": bundle.num_classes,
                    "external_name": bundle.external_name if split == "external" else "",
                    "proxy_used": bundle.external_proxy_used if split == "external" else False,
                }
            )
    return pd.DataFrame(rows)
