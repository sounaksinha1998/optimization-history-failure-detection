"""Spectral features along the shared training-time checkpoint axis.

``R`` is indexed as (T, K, d_v) or (N, T, K, d_v) where T is the common
checkpoint grid t_1 < ... < t_T. When ``use_fft=True``, ``np.fft.rfft`` runs
on that T axis only — not sample-visit order, batch index, or padded histories.
When ``use_fft=False``, tokens are the raw R flattened over (K, T).
There is no spectral amplification or learned frequency weighting.
"""

from __future__ import annotations

import numpy as np


def _time_axis(ndim: int) -> int:
    """Time axis index for (T, K, d_v) or (N, T, K, d_v)."""
    return 0 if ndim == 3 else 1


def _validate_trajectory_shape(R: np.ndarray) -> tuple[int, int, int, int | None]:
    arr = np.asarray(R, dtype=np.float32)
    if arr.ndim == 3:
        t_steps, num_levels, d_v = arr.shape
        return t_steps, num_levels, d_v, None
    if arr.ndim == 4:
        batch, t_steps, num_levels, d_v = arr.shape
        return t_steps, num_levels, d_v, batch
    raise ValueError(f"R must have shape (T, K, d_v) or (N, T, K, d_v), got {arr.shape}")


def temporal_fft_features(
    R: np.ndarray,
    *,
    use_fft: bool = True,
    lengths: np.ndarray | None = None,
) -> np.ndarray:
    """Convert trajectories to spectral or raw token features.

    Parameters
    ----------
    R
        Trajectory array with shape (T, K, d_v) for one sample or
        (N, T, K, d_v) for a batch.
    use_fft
        When ``True``, apply a real FFT along the training-time axis and
        return magnitude spectra. When ``False``, return the raw trajectory
        reshaped to token form.
    lengths
        Unused unless provided. All samples must share the same T. Passing
        per-sample lengths that differ from T is an error (no zero-padding).

    Returns
    -------
    np.ndarray
        For a single trajectory: shape (K * T_tokens, d_v).
        For a batch: shape (N, K * T_tokens, d_v).
        ``T_tokens`` equals ``T`` when ``use_fft=False`` and ``T_freq`` when
        ``use_fft=True``.
    """
    arr = np.asarray(R, dtype=np.float32)
    t_steps, num_levels, d_v, batch = _validate_trajectory_shape(arr)

    if not use_fft:
        if batch is None:
            return arr.reshape(num_levels * t_steps, d_v)
        return arr.reshape(batch, num_levels * t_steps, d_v)

    if lengths is not None:
        length_arr = np.asarray(lengths).reshape(-1)
        if np.any(length_arr != t_steps):
            raise ValueError(
                "All trajectories must share checkpoint length T="
                f"{t_steps}; got lengths={length_arr.tolist()}. "
                "Do not pad or pack per-sample visit histories."
            )
    time_axis = _time_axis(arr.ndim)
    work = arr

    if batch is None:
        spectrum = np.abs(np.fft.rfft(work, axis=time_axis))
        t_freq = spectrum.shape[0]
        return spectrum.reshape(num_levels * t_freq, d_v)

    spectrum = np.abs(np.fft.rfft(work, axis=time_axis))
    t_freq = spectrum.shape[1]
    return spectrum.reshape(batch, num_levels * t_freq, d_v)


def spectral_token_dim(
    t_steps: int,
    num_levels: int,
    *,
    use_fft: bool = True,
) -> int:
    """Number of memory tokens produced by :func:`temporal_fft_features`."""
    if not use_fft:
        return num_levels * t_steps
    t_freq = t_steps // 2 + 1
    return num_levels * t_freq
