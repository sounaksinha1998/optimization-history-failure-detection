"""Per-sample MSA alignments and memory-derived signals (phases 2–3)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from optimizer.associative_memory import (
    AssociativeMemoryConfig,
)
from optimizer.msa import compute_memory_signals, softmax_attention
from research.common.associative_attention import retrieve_z_memory_from_trajectory
from research.common.trajectory_store import (
    MemoryCheckpointBundle,
    MissingMemoryCheckpointsError,
    build_query_trajectory,
)


def _flatten_tree(tree: Any) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.zeros((0,), dtype=jnp.float32)
    return jnp.concatenate([x.reshape(-1) for x in leaves])


def tree_normalized_dot(a_tree: Any, b_tree: Any, eps: float) -> jnp.ndarray:
    """⟨â, b̂⟩ with â = a / (‖a‖ + ε), b̂ = b / (‖b‖ + ε)."""
    a_flat = _flatten_tree(a_tree)
    b_flat = _flatten_tree(b_tree)
    a_hat = a_flat / (jnp.linalg.norm(a_flat) + eps)
    b_hat = b_flat / (jnp.linalg.norm(b_flat) + eps)
    return jnp.dot(a_hat, b_hat)


def gradient_memory_alignments(
    grad_tree: Any,
    long_term: tuple[Any, ...],
    *,
    eps: float = 1e-8,
) -> tuple[jnp.ndarray, ...]:
    """c^(j) = ⟨ĝ, L̂^(j)⟩ for each long-term memory level."""
    return tuple(tree_normalized_dot(grad_tree, level, eps) for level in long_term)


def attention_from_alignments(
    alignments: tuple[jnp.ndarray, ...],
    *,
    tau: float = 1.0,
) -> tuple[jnp.ndarray, ...]:
    """α_j = softmax_j(c_j / τ)."""
    if not alignments:
        return ()
    c_stack = jnp.stack(alignments)
    alpha = softmax_attention(c_stack, tau)
    return tuple(alpha[j] for j in range(len(alignments)))


def per_sample_msa_signals(
    grad_tree: Any,
    long_term: tuple[Any, ...],
    *,
    tau: float = 1.0,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Compute alignments, attention, and Phase-3 memory signals for one example."""
    alignments = gradient_memory_alignments(grad_tree, long_term, eps=eps)
    alpha = attention_from_alignments(alignments, tau=tau)
    signals = compute_memory_signals(alignments, alpha, eps=eps)

    out: dict[str, float] = {}
    for j, c_j in enumerate(alignments, start=1):
        out[f"c_{j}"] = float(c_j)
    for j, a_j in enumerate(alpha, start=1):
        out[f"alpha_{j}"] = float(a_j)
    out["memory_agreement"] = float(signals.agreement)
    out["memory_novelty"] = float(signals.novelty)
    out["memory_disagreement"] = float(signals.disagreement)
    out["msa_entropy"] = float(signals.msa_entropy)
    return out


def per_sample_associative_signals(
    h: jnp.ndarray | np.ndarray,
    trajectory: np.ndarray,
    cfg: AssociativeMemoryConfig | None = None,
    *,
    attention_params: dict[str, jnp.ndarray] | None = None,
    d_out: int = 128,
    seed: int = 0,
) -> dict[str, float]:
    """Label-free memory retrieval from frozen ``h_T(x)`` and replayed ``R(x)``.

    ``trajectory`` must already be R[t,j] = M_t^j k_query with k_query = normalize(h_T).
    """
    resolved_cfg = cfg or AssociativeMemoryConfig()
    z, attn = retrieve_z_memory_from_trajectory(
        np.asarray(h, dtype=np.float32),
        np.asarray(trajectory, dtype=np.float32),
        cfg=resolved_cfg,
        attention_params=attention_params,
        d_out=d_out,
        seed=seed,
    )
    z_vec = np.asarray(z, dtype=np.float32).reshape(-1)
    attn_vec = np.asarray(attn, dtype=np.float32).reshape(-1)
    out: dict[str, float] = {
        f"z_{j}": float(z_vec[j]) for j in range(z_vec.shape[0])
    }
    out["z_memory_norm"] = float(np.linalg.norm(z_vec))
    if attn_vec.size:
        out["z_attn_entropy"] = float(
            -np.sum(attn_vec * np.log(np.clip(attn_vec, 1e-12, 1.0)))
        )
    return out


def batch_associative_z_memory(
    h: np.ndarray,
    checkpoints: MemoryCheckpointBundle,
    cfg: AssociativeMemoryConfig | None = None,
    *,
    sample_ids: np.ndarray | None = None,
    attention_params: dict[str, jnp.ndarray] | None = None,
    d_out: int = 128,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch label-free ``z_memory`` from frozen h_T and the shared checkpoint grid.

    ``sample_ids`` is ignored. Trajectories are not looked up by ID. Every sample
    is replayed as R[t,j] = M_t^j normalize(h_T). Missing checkpoints are an error,
    not a T=1 fallback.
    """
    del sample_ids
    resolved_cfg = cfg or AssociativeMemoryConfig()
    if checkpoints.T < 1:
        raise MissingMemoryCheckpointsError(
            "No associative memory checkpoints available. Cannot fall back to T=1."
        )
    h_arr = np.asarray(h, dtype=np.float32)
    if h_arr.ndim == 1:
        h_arr = h_arr[None, :]
    R = build_query_trajectory(h_arr, checkpoints, eps=resolved_cfg.eps)
    pool_seed = resolved_cfg.aggregation_seed if seed is None else seed
    z, attn = retrieve_z_memory_from_trajectory(
        h_arr,
        R,
        cfg=resolved_cfg,
        attention_params=attention_params,
        d_out=d_out,
        seed=pool_seed,
    )
    z = np.asarray(z, dtype=np.float32)
    attn = np.asarray(attn, dtype=np.float32)
    if z.ndim == 1:
        z = z[None, :]
    if attn.ndim == 1:
        attn = attn[None, :]
    return z, attn


def example_grad(params: dict[str, jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray) -> Any:
    """Gradient of single-example cross-entropy w.r.t. model parameters."""

    def loss_fn(p: dict[str, jnp.ndarray]) -> jnp.ndarray:
        logit_vec = _mlp_logits(p, x)
        return optax.softmax_cross_entropy_with_integer_labels(logit_vec, y)

    return jax.grad(loss_fn)(params)


def _mlp_logits(params: dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    x_flat = x.reshape(-1)
    h1 = jax.nn.relu(x_flat @ params["w1"] + params["b1"])
    h2 = jax.nn.relu(h1 @ params["w2"] + params["b2"])
    return h2 @ params["w3"] + params["b3"]
