"""NRO-NRMv2 research optimizer — memory observation and directional sync.

Equations (per parameter tensor; ⊙ is elementwise product)
----------------------------------------------------------

**Nested fast / medium / slow memories** (λ_γ > λ_β > λ_α):

    G_t = (1 - λ_γ) G_{t-1} + λ_γ g_t
    B_t = (1 - λ_β) B_{t-1} + λ_β G_t
    A_t = (1 - λ_α) A_{t-1} + λ_α B_t

with λ(τ) = 1 - exp(-1/τ) and τ_β = φ τ_γ, τ_α = φ² τ_γ (φ = golden ratio).

**Preconditioned direction M^(0)** (optional residual shaping):

    R_GB = (G_t - B_t) / (|G_t| + |B_t| + ε)
    R_BA = (B_t - A_t) / (|B_t| + |A_t| + ε)
    M^(0) = G_t ⊙ (1 + c_β R_GB + c_α R_BA)

**Hierarchical long-term memories** (K levels, λ_1 > λ_2 > … > λ_K):

    L^(1)_t = (1 - λ_1) L^(1)_{t-1} + λ_1 A_t
    L^(j)_t = (1 - λ_j) L^(j)_{t-1} + λ_j L^(j-1)_t,   j ≥ 2

**Directional synchronization** (`research_nrm_v2`, per tensor leaf):

    M̂^(0) = M^(0) / (‖M^(0)‖ + ε)
    L̂^(j) = L^(j) / (‖L^(j)‖ + ε)
    c_k   = ⟨M̂^(0), L̂^(k)⟩
    q_k   = max(0, c_k)
    ρ     = ρ_max Σ_k w_k q_k
    M_sync = ‖M^(0)‖ · [(1 - ρ) M̂^(0) + ρ Σ_k w_k L̂^(k)]

**D4 denominator** (default):

    D_t = √|A_t| + ε
    u_t = M_sync / D_t        (full NRM v2)
    θ_{t+1} = θ_t - η u_t

**Observe-only mode** (`research_nrm_v2_observe`):

    Memories G_t, B_t, A_t, L^(j)_t are updated each step, but parameters follow
    standard Adam on raw gradients g_t (memory does not steer the update).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from optax._src import base

from optimizer.core import memory_preconditioned_step
from optimizer.memory import init_long_term_zeros, long_term_accumulation_step
from optimizer.timescales import compute_long_term_lambdas, compute_timescale_lambdas, default_timescale_ratio

Params = base.Params
Updates = base.Updates
ScalarOrSchedule = base.ScalarOrSchedule


class NRMv2State(NamedTuple):
    """Optimizer state for NRM v2 research skeleton."""

    gamma: Any
    beta: Any
    alpha: Any
    v: Any
    long_term: tuple[Any, ...]
    step: jnp.ndarray


def _tensor_l2_norm(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jnp.sum(jnp.square(x)))


def _sync_single_tensor(
    m0_l: jnp.ndarray,
    level_leaves: tuple[jnp.ndarray, ...],
    weights: tuple[float, ...],
    rho_max: float,
    eps: float,
) -> jnp.ndarray:
    """Per-tensor directional sync returning synchronized M^(n,l)."""
    s_l = _tensor_l2_norm(m0_l)
    m0_hat = m0_l / (s_l + eps)
    rho_l = jnp.array(0.0, dtype=m0_l.dtype)
    weighted_lt = jnp.zeros_like(m0_l)

    for w_k, level_l in zip(weights, level_leaves):
        l_hat = level_l / (_tensor_l2_norm(level_l) + eps)
        c_k = jnp.sum(m0_hat * l_hat)
        q_k = jnp.maximum(jnp.array(0.0, dtype=m0_l.dtype), c_k)
        rho_l = rho_l + w_k * q_k
        weighted_lt = weighted_lt + w_k * l_hat

    rho_l = jnp.asarray(rho_max, dtype=m0_l.dtype) * rho_l
    m_hat_n = (1.0 - rho_l) * m0_hat + rho_l * weighted_lt
    return s_l * m_hat_n


def apply_directional_sync(
    m0: Any,
    long_term_levels: tuple[Any, ...],
    weights: tuple[float, ...],
    rho_max: float,
    eps: float,
) -> Any:
    """Apply per-tensor NRM v2 directional synchronization."""
    if not long_term_levels or rho_max == 0.0:
        return m0

    def sync_leaf(m0_l: jnp.ndarray, *level_leaves: jnp.ndarray) -> jnp.ndarray:
        return _sync_single_tensor(m0_l, level_leaves, weights, rho_max, eps)

    return jax.tree_util.tree_map(
        sync_leaf,
        m0,
        *long_term_levels,
        is_leaf=lambda x: isinstance(x, jnp.ndarray),
    )


def resolve_sync_weights(depth: int, weights: tuple[float, ...] | None = None) -> tuple[float, ...]:
    if depth <= 0:
        return ()
    if weights is None:
        return tuple(1.0 / depth for _ in range(depth))
    if len(weights) != depth:
        raise ValueError(f"sync_weights length {len(weights)} must match depth {depth}")
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("sync_weights must sum to a positive value")
    return tuple(float(w) / total for w in weights)


def research_nrm_v2_observe(
    learning_rate: ScalarOrSchedule,
    *,
    tau: float = 8.0,
    timescale_ratio: float | None = None,
    long_term_depth: int = 2,
    long_term_timescale_multiplier: float | None = None,
    eps: float = 1e-8,
    adam_beta: float = 0.999,
) -> base.GradientTransformation:
    """Phase-1 skeleton: maintain NRM v2 memory but apply Adam updates only.

    Long-term memories L^(j) are updated from A_t each step. The parameter
    update uses standard Adam on raw gradients — memory does not steer training.
  """
    ratio = timescale_ratio if timescale_ratio is not None else default_timescale_ratio()
    lambdas = compute_timescale_lambdas(tau, ratio)
    lt_lambdas = compute_long_term_lambdas(
        tau, ratio, long_term_depth, long_term_timescale_multiplier
    )
    adam_tx = optax.adam(learning_rate, b1=0.9, b2=adam_beta, eps=eps)

    def init_fn(params: Params) -> tuple[NRMv2State, Any]:
        zeros = jax.tree_util.tree_map(lambda p: jnp.zeros(p.shape, dtype=jnp.float32), params)
        return (
            NRMv2State(
                gamma=zeros,
                beta=zeros,
                alpha=zeros,
                v=zeros,
                long_term=init_long_term_zeros(params, long_term_depth),
                step=jnp.array(0, dtype=jnp.int32),
            ),
            adam_tx.init(params),
        )

    def update_fn(
        grads: Updates, state: tuple[NRMv2State, Any], params: Params | None = None
    ) -> tuple[Updates, tuple[NRMv2State, Any]]:
        mem_state, adam_state = state
        step = mem_state.step + 1
        warm_start = mem_state.step == 0

        mem = memory_preconditioned_step(
            grads,
            mem_state.gamma,
            mem_state.beta,
            mem_state.alpha,
            mem_state.v,
            lambdas["lambda_gamma"],
            lambdas["lambda_beta"],
            lambdas["lambda_alpha"],
            eps,
            warm_start=warm_start,
            denominator="D4",
        )
        new_long_term = long_term_accumulation_step(mem.alpha, mem_state.long_term, lt_lambdas)

        updates, new_adam_state = adam_tx.update(grads, adam_state, params)
        new_mem_state = NRMv2State(
            gamma=mem.gamma,
            beta=mem.beta,
            alpha=mem.alpha,
            v=mem_state.v,
            long_term=new_long_term,
            step=step,
        )
        return updates, (new_mem_state, new_adam_state)

    return base.GradientTransformation(init_fn, update_fn)


def research_nrm_v2(
    learning_rate: ScalarOrSchedule,
    *,
    tau: float = 8.0,
    timescale_ratio: float | None = None,
    residual_beta: float = 0.0,
    residual_alpha: float = 0.0,
    long_term_depth: int = 2,
    long_term_timescale_multiplier: float | None = None,
    sync_rho_max: float = 0.25,
    sync_weights: tuple[float, ...] | None = None,
    eps: float = 1e-8,
    denominator_beta_v: float = 0.999,
) -> base.GradientTransformation:
    """Full NRM v2 update: M^(0) with optional directional sync and D4 denominator."""
    ratio = timescale_ratio if timescale_ratio is not None else default_timescale_ratio()
    lambdas = compute_timescale_lambdas(tau, ratio)
    lt_lambdas = compute_long_term_lambdas(
        tau, ratio, long_term_depth, long_term_timescale_multiplier
    )
    weights = resolve_sync_weights(long_term_depth, sync_weights)

    def init_fn(params: Params) -> NRMv2State:
        zeros = jax.tree_util.tree_map(lambda p: jnp.zeros(p.shape, dtype=jnp.float32), params)
        return NRMv2State(
            gamma=zeros,
            beta=zeros,
            alpha=zeros,
            v=zeros,
            long_term=init_long_term_zeros(params, long_term_depth),
            step=jnp.array(0, dtype=jnp.int32),
        )

    def update_fn(
        grads: Updates, state: NRMv2State, params: Params | None = None
    ) -> tuple[Updates, NRMv2State]:
        del params
        step = state.step + 1
        warm_start = state.step == 0

        mem = memory_preconditioned_step(
            grads,
            state.gamma,
            state.beta,
            state.alpha,
            state.v,
            lambdas["lambda_gamma"],
            lambdas["lambda_beta"],
            lambdas["lambda_alpha"],
            eps,
            warm_start=warm_start,
            residual_beta=residual_beta,
            residual_alpha=residual_alpha,
            denominator="D4",
            denominator_beta_v=denominator_beta_v,
        )
        new_long_term = long_term_accumulation_step(mem.alpha, state.long_term, lt_lambdas)
        numerator = apply_directional_sync(
            mem.combined_direction,
            new_long_term,
            weights,
            sync_rho_max,
            eps,
        )
        direction = jax.tree_util.tree_map(lambda m, d: m / d, numerator, mem.denominator)

        lr = (
            learning_rate(step)
            if callable(learning_rate)
            else jnp.asarray(learning_rate, dtype=jnp.float32)
        )
        updates = jax.tree_util.tree_map(lambda d: -lr * d, direction)
        new_state = NRMv2State(
            gamma=mem.gamma,
            beta=mem.beta,
            alpha=mem.alpha,
            v=mem.v,
            long_term=new_long_term,
            step=step,
        )
        return updates, new_state

    return base.GradientTransformation(init_fn, update_fn)
