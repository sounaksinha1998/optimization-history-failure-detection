"""Label-free retrieval of z_memory from penultimate features and spectral tokens."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from optimizer.associative_memory import AssociativeMemoryConfig


def tokens_to_matrix(S: np.ndarray) -> np.ndarray:
    """Normalize spectral/raw tokens to a rank-2 token matrix.

    Accepts shape (num_tokens, d_v) or (batch, num_tokens, d_v).
    """
    arr = np.asarray(S, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr
    raise ValueError(f"S must have shape (num_tokens, d_v) or (batch, num_tokens, d_v), got {arr.shape}")


def _fixed_token_projection(
    tokens: np.ndarray,
    *,
    d_out: int,
    seed: int,
) -> np.ndarray:
    """Deterministic Gaussian projection from token features to d_out."""
    tokens = np.asarray(tokens, dtype=np.float32)
    if tokens.ndim == 1:
        tokens = tokens[None, :]
    in_dim = tokens.shape[-1]
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((in_dim, d_out)).astype(np.float32)
    proj /= np.sqrt(in_dim).astype(np.float32)
    return tokens @ proj


def mean_pool_tokens(
    S: np.ndarray,
    *,
    d_out: int,
    seed: int = 0,
) -> np.ndarray:
    """Mean-pool token sequences into a fixed-size vector."""
    tokens = tokens_to_matrix(S)
    if tokens.ndim == 2:
        pooled = tokens.mean(axis=0, keepdims=True)
        return _fixed_token_projection(pooled, d_out=d_out, seed=seed).reshape(-1)
    pooled = tokens.mean(axis=1)
    return _fixed_token_projection(pooled, d_out=d_out, seed=seed)


def _init_simple_attention_params(
    key: jax.Array,
    *,
    h_dim: int,
    token_dim: int,
    d_out: int,
) -> dict[str, jnp.ndarray]:
  keys = jax.random.split(key, 3)
  scale = 0.01
  return {
      "W_Q": jax.random.normal(keys[0], (h_dim, d_out)) * scale,
      "W_K": jax.random.normal(keys[1], (token_dim, d_out)) * scale,
      "W_V": jax.random.normal(keys[2], (token_dim, d_out)) * scale,
  }


def simple_attention_forward(
    h: jnp.ndarray,
    tokens: jnp.ndarray,
    params: dict[str, jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Scaled dot-product attention from h over token matrix.

    Parameters
    ----------
    h
        Query features with shape (d_h,) or (batch, d_h).
    tokens
        Token matrix with shape (num_tokens, d_v) or (batch, num_tokens, d_v).
    """
    if h.ndim == 1:
        h = h[None, :]
    if tokens.ndim == 2:
        tokens = tokens[None, :]

    q = h @ params["W_Q"]
    k = tokens @ params["W_K"]
    v = tokens @ params["W_V"]
    scale = jnp.sqrt(jnp.asarray(q.shape[-1], dtype=jnp.float32))
    logits = jnp.einsum("bd,bsd->bs", q, k) / scale
    attn = jax.nn.softmax(logits, axis=-1)
    z = jnp.einsum("bs,bsd->bd", attn, v)
    return z.reshape(-1) if z.shape[0] == 1 else z, attn


def msa_attention_forward_batch(
    h_batch: jnp.ndarray,
    token_batch: jnp.ndarray,
    params: dict[str, jnp.ndarray],
    *,
    n_heads: int,
    d_k: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Multi-head attention compatible with trained MSA parameters."""
    if h_batch.ndim == 1:
        h_batch = h_batch[None, :]
    if token_batch.ndim == 2:
        token_batch = token_batch[None, :]

    z_heads: list[jnp.ndarray] = []
    attn_heads: list[jnp.ndarray] = []
    scale = jnp.sqrt(jnp.asarray(d_k, dtype=jnp.float32))
    for head in range(n_heads):
        q = h_batch @ params[f"W_Q_{head}"]
        k = token_batch @ params[f"W_K_{head}"]
        v = token_batch @ params[f"W_V_{head}"]
        logits = jnp.einsum("bd,bsd->bs", q, k) / scale
        attn = jax.nn.softmax(logits, axis=-1)
        z_heads.append(jnp.einsum("bs,bsd->bd", attn, v))
        attn_heads.append(attn)
    z_cat = jnp.concatenate(z_heads, axis=-1)
    z_out = z_cat @ params["W_O"]
    attn = jnp.stack(attn_heads, axis=1)
    return z_out, attn


def retrieve_z_memory(
    h: np.ndarray,
    S: np.ndarray,
    *,
    use_attention: bool | None = None,
    attention_params: dict[str, jnp.ndarray] | None = None,
    cfg: AssociativeMemoryConfig | None = None,
    d_out: int = 128,
    n_heads: int = 4,
    d_k: int = 64,
    seed: int | None = None,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve label-free z_memory from h_T(x) and spectral/raw tokens.

    Primary path (``use_attention=False``): deterministic mean-pool of S plus a
    fixed Gaussian projection. This is not learned attention.

    Optional path (``use_attention=True``): scaled-dot-product attention with
    frozen ``attention_params`` trained on calibration only.
    """
    resolved_cfg = cfg or AssociativeMemoryConfig()
    if use_attention is None:
        use_attention = resolved_cfg.use_attention

    tokens = tokens_to_matrix(S)
    h_arr = np.asarray(h, dtype=np.float32)
    pool_seed = resolved_cfg.aggregation_seed if seed is None else int(seed)

    if not use_attention:
        if tokens.ndim == 2:
            z = mean_pool_tokens(tokens, d_out=d_out, seed=pool_seed)
            uniform = np.ones((1, tokens.shape[0]), dtype=np.float32) / tokens.shape[0]
            return np.asarray(z, dtype=np.float32), uniform
        z_parts: list[np.ndarray] = []
        attn_parts: list[np.ndarray] = []
        for row in tokens:
            z_row = mean_pool_tokens(row, d_out=d_out, seed=pool_seed)
            z_parts.append(z_row)
            attn_parts.append(np.full((row.shape[0],), 1.0 / row.shape[0], dtype=np.float32))
        return np.stack(z_parts, axis=0), np.stack(attn_parts, axis=0)

    if attention_params is None or "W_O" not in attention_params:
        raise ValueError(
            "use_attention=True requires frozen attention_params trained on the "
            "calibration split. Random WQ/WK/WV and sample-ID seeds are not allowed. "
            "For the primary experiment set use_attention=False (mean/raw pooling)."
        )

    h_j = jnp.asarray(h_arr, dtype=jnp.float32)
    token_j = jnp.asarray(tokens, dtype=jnp.float32)
    z_parts = []
    attn_parts = []

    if h_j.ndim == 1:
        tb = token_j if token_j.ndim == 2 else token_j[0]
        zb, ab = msa_attention_forward_batch(
            h_j,
            tb,
            attention_params,
            n_heads=n_heads,
            d_k=d_k,
        )
        return np.asarray(zb.reshape(-1), dtype=np.float32), np.asarray(
            ab.reshape(ab.shape[1], ab.shape[2]), dtype=np.float32
        )
    for start in range(0, h_j.shape[0], batch_size):
        hb = h_j[start : start + batch_size]
        tb = token_j if token_j.ndim == 2 else token_j[start : start + batch_size]
        zb, ab = msa_attention_forward_batch(
            hb,
            tb,
            attention_params,
            n_heads=n_heads,
            d_k=d_k,
        )
        z_parts.append(np.asarray(zb))
        attn_parts.append(np.asarray(ab))
    return np.concatenate(z_parts, axis=0), np.concatenate(attn_parts, axis=0)


def retrieve_z_memory_from_trajectory(
    h: np.ndarray,
    R: np.ndarray,
    *,
    use_fft: bool | None = None,
    use_attention: bool | None = None,
    attention_params: dict[str, jnp.ndarray] | None = None,
    cfg: AssociativeMemoryConfig | None = None,
    lengths: np.ndarray | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: frozen-query trajectory -> spectral tokens -> z_memory."""
    from optimizer.associative_memory import assert_finite_array
    from research.common.spectral_memory import temporal_fft_features

    resolved_cfg = cfg or AssociativeMemoryConfig()
    fft_enabled = resolved_cfg.use_fft if use_fft is None else use_fft
    R = assert_finite_array(np.asarray(R, dtype=np.float32), name="trajectory R")
    S = temporal_fft_features(R, use_fft=fft_enabled, lengths=lengths)
    return retrieve_z_memory(
        h,
        S,
        use_attention=use_attention,
        attention_params=attention_params,
        cfg=resolved_cfg,
        **kwargs,
    )
