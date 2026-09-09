"""Multi-scale attention (MSA) math and memory-derived signals.

Equations (per parameter tensor leaf)
-------------------------------------

**Normalized direction and long-term memories**:

    M̂^(0) = M^(0) / (‖M^(0)‖ + ε)
    L̂^(j) = L^(j) / (‖L^(j)‖ + ε)

**Gradient–memory alignments**:

    c_j = ⟨M̂^(0), L̂^(j)⟩

**Per-timescale second moments** (j = 1 … K):

    V^(j)_t = (1 - λ_j) V^(j)_{t-1} + λ_j g_t²
    V̂^(j)_t = V^(j)_t / (1 - (1 - λ_j)^t)          (bias correction)

**Adam second moment** (mode ``adam_d``):

    V_adam,t = β V_adam,t-1 + (1 - β) g_t²
    V̂_adam,t = V_adam,t / (1 - β^t)

**MSA attention weights**:

    α_j = softmax_j(c_j / τ)

**Blended denominator Ṽ** (mode-dependent):

    adam_d:     Ṽ = V̂_adam
    uniform_ms: Ṽ = (1/K) Σ_j V̂^(j)
    msa:        Ṽ = Σ_j α_j V̂^(j)

**Preconditioned update direction**:

    u_t = M^(0) / (√Ṽ + ε)

**Memory-derived diagnostic signals**:

    A     = Σ_j α_j c_j                         (agreement)
    N     = 1 - A                               (novelty)
    D     = Σ_j α_j (c_j - A)²                  (cross-timescale disagreement)
    H^MSA = -Σ_j α_j log α_j                    (MSA entropy)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp

MSAMode = Literal["v5b", "adam_d", "uniform_ms", "msa"]
MSA_MODES: tuple[MSAMode, ...] = ("v5b", "adam_d", "uniform_ms", "msa")


def tensor_l2_norm(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jnp.sum(jnp.square(x)))


def normalize_tensor(x: jnp.ndarray, eps: float) -> jnp.ndarray:
    return x / (tensor_l2_norm(x) + eps)


def bias_correct_scale_moment(v: jnp.ndarray, step: jnp.ndarray, lam: jnp.ndarray) -> jnp.ndarray:
    decay = 1.0 - lam
    correction = 1.0 - jnp.power(decay, step)
    return v / jnp.maximum(correction, jnp.asarray(1e-12, dtype=v.dtype))


def attention_entropy(alpha: jnp.ndarray, eps: float) -> jnp.ndarray:
    safe = jnp.maximum(alpha, eps)
    return -jnp.sum(alpha * jnp.log(safe))


def softmax_attention(c: jnp.ndarray, tau: float) -> jnp.ndarray:
    tau_f = jnp.asarray(tau, dtype=c.dtype)
    logits = c / tau_f
    logits = logits - jnp.max(logits)
    exp_logits = jnp.exp(logits)
    return exp_logits / jnp.sum(exp_logits)


@dataclass(frozen=True)
class MemorySignals:
    """Phase-3 memory-derived scalar signals (per tensor leaf or aggregated)."""

    agreement: jnp.ndarray
    novelty: jnp.ndarray
    disagreement: jnp.ndarray
    msa_entropy: jnp.ndarray
    alignments: tuple[jnp.ndarray, ...]
    alpha: tuple[jnp.ndarray, ...]


def compute_memory_signals(
    alignments: tuple[jnp.ndarray, ...],
    alpha: tuple[jnp.ndarray, ...],
    eps: float = 1e-8,
) -> MemorySignals:
    """Compute agreement, novelty, disagreement, and MSA entropy from c_j and α_j."""
    if not alignments or not alpha:
        zero = jnp.array(0.0, dtype=jnp.float32)
        return MemorySignals(
            agreement=zero,
            novelty=zero,
            disagreement=zero,
            msa_entropy=zero,
            alignments=(),
            alpha=(),
        )

    c_stack = jnp.stack(alignments)
    alpha_stack = jnp.stack(alpha)
    agreement = jnp.sum(alpha_stack * c_stack)
    novelty = 1.0 - agreement
    disagreement = jnp.sum(alpha_stack * jnp.square(c_stack - agreement))
    entropy = attention_entropy(alpha_stack, eps)
    return MemorySignals(
        agreement=agreement,
        novelty=novelty,
        disagreement=disagreement,
        msa_entropy=entropy,
        alignments=alignments,
        alpha=alpha,
    )


def compute_leaf_alignments(
    m0: jnp.ndarray,
    long_term_levels: tuple[jnp.ndarray, ...],
    eps: float,
) -> tuple[jnp.ndarray, ...]:
    """c^(j) = <M̂^(0), L̂^(j)> for each timescale."""
    m0_hat = normalize_tensor(m0, eps)
    alignments: list[jnp.ndarray] = []
    for level in long_term_levels:
        l_hat = normalize_tensor(level, eps)
        alignments.append(jnp.sum(m0_hat * l_hat))
    return tuple(alignments)


def update_scale_moments(
    grad: jnp.ndarray,
    v_levels: tuple[jnp.ndarray, ...],
    lambdas: tuple[jnp.ndarray, ...],
    step: jnp.ndarray,
) -> tuple[tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...]]:
    """Update and bias-correct per-timescale second moments V^(j)."""
    new_levels: list[jnp.ndarray] = []
    v_hat_list: list[jnp.ndarray] = []
    for v_j, lam_j in zip(v_levels, lambdas):
        lam = jnp.asarray(lam_j, dtype=grad.dtype)
        v_new = (1.0 - lam) * v_j + lam * jnp.square(grad)
        v_hat = bias_correct_scale_moment(v_new, step, lam)
        new_levels.append(v_new)
        v_hat_list.append(v_hat)
    return tuple(new_levels), tuple(v_hat_list)


def blend_denominator(
    mode: MSAMode,
    v_hat_list: tuple[jnp.ndarray, ...],
    alignments: tuple[jnp.ndarray, ...],
    adam_v_hat: jnp.ndarray,
    tau: float,
    eps: float,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...], jnp.ndarray, jnp.ndarray]:
    """Return Ṽ, per-level α_j scalars, entropy, dominant scale index."""
    depth = len(v_hat_list)

    if mode == "adam_d" or depth == 0:
        if depth == 0:
            alpha_levels: tuple[jnp.ndarray, ...] = ()
        else:
            scalar = jnp.array(1.0 / depth, dtype=adam_v_hat.dtype)
            alpha_levels = tuple(scalar for _ in range(depth))
        return adam_v_hat, alpha_levels, jnp.array(0.0, dtype=adam_v_hat.dtype), jnp.array(0.0, dtype=adam_v_hat.dtype)

    if mode == "uniform_ms":
        v_stack = jnp.stack(v_hat_list)
        v_tilde = jnp.mean(v_stack, axis=0)
        scalar = jnp.array(1.0 / depth, dtype=adam_v_hat.dtype)
        alpha_levels = tuple(scalar for _ in range(depth))
        entropy = attention_entropy(jnp.stack(alpha_levels), eps)
        return v_tilde, alpha_levels, entropy, jnp.array(0.0, dtype=adam_v_hat.dtype)

    c_stack = jnp.stack(alignments)
    alpha_arr = softmax_attention(c_stack, tau)
    v_stack = jnp.stack(v_hat_list)
    v_tilde = jnp.einsum("j,j...->...", alpha_arr, v_stack)
    alpha_levels = tuple(alpha_arr[j] for j in range(depth))
    entropy = attention_entropy(alpha_arr, eps)
    dominant = jnp.argmax(alpha_arr).astype(adam_v_hat.dtype)
    return v_tilde, alpha_levels, entropy, dominant


def msa_direction(
    m0: jnp.ndarray,
    grad: jnp.ndarray,
    long_term_levels: tuple[jnp.ndarray, ...],
    v_levels: tuple[jnp.ndarray, ...],
    adam_v: jnp.ndarray,
    step: jnp.ndarray,
    lambdas: tuple[float, ...],
    mode: MSAMode,
    tau: float,
    eps: float,
    adam_beta: float,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...], jnp.ndarray, tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...], MemorySignals, jnp.ndarray]:
    """Compute MSA denominator direction u ∝ M^(0) / sqrt(Ṽ) and diagnostics."""
    lam_arrays = tuple(jnp.asarray(lam, dtype=m0.dtype) for lam in lambdas)
    alignments = compute_leaf_alignments(m0, long_term_levels, eps)

    new_v_levels, v_hat_list = update_scale_moments(grad, v_levels, lam_arrays, step)
    new_adam_v = jnp.asarray(adam_beta, dtype=m0.dtype) * adam_v + (
        1.0 - jnp.asarray(adam_beta, dtype=m0.dtype)
    ) * jnp.square(grad)
    adam_correction = 1.0 - jnp.power(jnp.asarray(adam_beta, dtype=m0.dtype), step)
    adam_v_hat = new_adam_v / jnp.maximum(adam_correction, jnp.asarray(1e-12, dtype=m0.dtype))

    v_tilde, alpha_levels, entropy, dominant = blend_denominator(
        mode, v_hat_list, alignments, adam_v_hat, tau, eps
    )
    denom = jnp.sqrt(jnp.maximum(v_tilde, jnp.array(0.0, dtype=m0.dtype))) + eps
    direction = m0 / denom

    signals = compute_memory_signals(alignments, alpha_levels, eps)
    return (
        direction,
        new_v_levels,
        new_adam_v,
        alignments,
        alpha_levels,
        signals,
        dominant,
    )
