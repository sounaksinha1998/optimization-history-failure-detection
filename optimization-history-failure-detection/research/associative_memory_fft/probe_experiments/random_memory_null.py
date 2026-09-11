"""Null check: does learned memory M add signal beyond a random linear map R?

Compare z_j = M^j k(x) with r_j = R^j k(x) where k = h_T / ||h_T|| and R^j has the
same shape as M^j but is drawn from a standard Gaussian (fixed per draw).

Evaluation is test-only and label-free: AUROC(-max_j ||response_j||), so no probe fitting.
Cached z magnitudes and recomputed M@k are cross-checked. Multiple random R seeds give a
null distribution (mean +/- std).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.associative_memory_fft.probe_experiments import probe_analysis as pa
from research.common.memory import load_associative_artifacts

PROBE_ROOT = Path(__file__).resolve().parent
SEEDS = pa.SEEDS
NUM_RANDOM_R = 50
R_BASE_SEED = 9001
TASK_OFFSET = {"dermamnist": 0, "bloodmnist": 10_000, "organcmnist": 20_000}


def _paths(task: str) -> tuple[Path, Path, Path]:
    cross = task != "dermamnist"
    if cross:
        artifact = REPO_ROOT / "research" / "associative_memory_fft" / "cross_dataset"
        feat = artifact / "probe_features" / task
        ht_cache = PROBE_ROOT / "ht_probe_cache" / task
    else:
        artifact = REPO_ROOT / "research" / "associative_memory_fft" / "artifacts"
        feat = PROBE_ROOT / "feature_cache"
        ht_cache = PROBE_ROOT / "ht_probe_cache"
    return artifact, feat, ht_cache


def _normalize_rows(h: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(h, axis=1, keepdims=True)
    return h / (norms + eps)


def _level_responses(matrices: tuple[np.ndarray, ...], k: np.ndarray) -> np.ndarray:
    """k: (n, d_k) -> (n, num_levels, d_v)."""
    n = k.shape[0]
    d_v = matrices[0].shape[0]
    out = np.empty((n, len(matrices), d_v), dtype=np.float64)
    for j, matrix in enumerate(matrices):
        out[:, j, :] = k @ np.asarray(matrix, dtype=np.float64).T
    return out


def _random_matrices(
    template: tuple[np.ndarray, ...],
    rng: np.random.Generator,
    *,
    match_frobenius: bool,
) -> tuple[np.ndarray, ...]:
    mats: list[np.ndarray] = []
    for m in template:
        r = rng.standard_normal(m.shape).astype(np.float64)
        if match_frobenius:
            fro_m = float(np.linalg.norm(m))
            fro_r = float(np.linalg.norm(r))
            if fro_r > 0:
                r *= fro_m / fro_r
        mats.append(r)
    return tuple(mats)


def _mag_score(responses: np.ndarray) -> np.ndarray:
    """Label-free failure score: higher => more likely error."""
    mags = np.linalg.norm(responses, axis=2)
    return -np.max(mags, axis=1)


def _auroc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def _load_test_frame(task: str, seed: int, feat_root: Path) -> pd.DataFrame:
    test_path = feat_root / f"seed{seed}" / "test_features_full.csv"
    test_full = pd.read_csv(test_path)
    return pa.subsample_test_df(test_full, sample_size=pa.TEST_SAMPLE_SIZE, seed=pa.RANDOM_SEED + seed)


def _load_ht(test_df: pd.DataFrame, ht_cache: Path, seed: int) -> np.ndarray:
    cache_dir = ht_cache / f"seed{seed}"
    ht = np.load(cache_dir / "test_ht.npy").astype(np.float64)
    cached_ids = np.load(cache_dir / "test_sample_ids.npy", allow_pickle=True).astype(str)
    id_to_row = {sid: i for i, sid in enumerate(cached_ids)}
    want_ids = test_df["sample_id"].astype(str).to_numpy()
    try:
        idx = np.array([id_to_row[sid] for sid in want_ids], dtype=int)
    except KeyError as exc:
        raise ValueError(f"h_T cache missing sample_id for seed {seed}: {exc}") from exc
    return ht[idx]


def run_task(
    task: str,
    *,
    num_random: int = NUM_RANDOM_R,
    match_frobenius: bool = False,
) -> dict[str, Any]:
    artifact_dir, feat_root, ht_cache = _paths(task)
    rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        test_df = _load_test_frame(task, seed, feat_root)
        h = _load_ht(test_df, ht_cache, seed)
        k = _normalize_rows(h)
        y = test_df["error"].to_numpy(dtype=int)

        mem, _, _ = load_associative_artifacts(artifact_dir, task, seed)
        m_mats = tuple(np.asarray(m, dtype=np.float64) for m in mem.matrices)
        z_mk = _level_responses(m_mats, k)

        cached_mag = test_df[[f"z{j}_magnitude" for j in range(1, 5)]].to_numpy(dtype=np.float64)
        recomputed_mag = np.linalg.norm(z_mk, axis=2)
        max_mag_diff = float(np.max(np.abs(cached_mag - recomputed_mag)))

        z_cache_score = _mag_score(cached_mag[:, :, None])
        z_mk_score = _mag_score(z_mk)

        r_rk_scores: list[float] = []
        for r_idx in range(num_random):
            rng = np.random.default_rng(R_BASE_SEED + TASK_OFFSET.get(task, 0) + seed + r_idx)
            r_mats = _random_matrices(m_mats, rng, match_frobenius=match_frobenius)
            r_rk = _level_responses(r_mats, k)
            r_rk_scores.append(_auroc(y, _mag_score(r_rk)))

        rows.append(
            {
                "task": task,
                "seed": seed,
                "n_test": int(len(y)),
                "max_abs_mag_diff_cache_vs_Mk": max_mag_diff,
                "auroc_Mk_cached_mag": _auroc(y, z_cache_score),
                "auroc_Mk_recomputed_mag": _auroc(y, z_mk_score),
                "auroc_Rk_mag_mean": float(np.mean(r_rk_scores)),
                "auroc_Rk_mag_std": float(np.std(r_rk_scores)),
            }
        )

    out_df = pd.DataFrame(rows)
    summary = {
        "task": task,
        "num_random_R": num_random,
        "match_frobenius": match_frobenius,
        "mean_auroc": {
            "M_k_cached_mag": float(out_df["auroc_Mk_cached_mag"].mean()),
            "M_k_recomputed_mag": float(out_df["auroc_Mk_recomputed_mag"].mean()),
            "R_k_mag": float(out_df["auroc_Rk_mag_mean"].mean()),
        },
        "delta_Mk_minus_Rk": float(
            out_df["auroc_Mk_recomputed_mag"].mean() - out_df["auroc_Rk_mag_mean"].mean()
        ),
        "per_seed": rows,
    }
    return summary


def main() -> None:
    tasks = [a for a in sys.argv[1:] if not a.startswith("-")] or ["dermamnist", "bloodmnist", "organcmnist"]
    match_fro = "--match-frobenius" in sys.argv
    out_dir = PROBE_ROOT / "ht_probe_results" / "random_memory_null"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict[str, Any]] = []
    print("Test-only label-free AUROC (-max_j ||response_j||); compare M@k vs R@k\n")
    for task in tasks:
        summary = run_task(task, match_frobenius=match_fro)
        all_summaries.append(summary)
        m = summary["mean_auroc"]
        print(f"=== {task} ===")
        print(f"  M@k (cached mag):     {m['M_k_cached_mag']:.4f}")
        print(f"  M@k (recomputed mag): {m['M_k_recomputed_mag']:.4f}")
        print(f"  R@k (null, mean):     {m['R_k_mag']:.4f}")
        print(f"  M@k - R@k:            {summary['delta_Mk_minus_Rk']:+.4f}")

    (out_dir / "random_memory_null.json").write_text(json.dumps(all_summaries, indent=2))
    print(f"\nWrote {out_dir / 'random_memory_null.json'}")


if __name__ == "__main__":
    main()
