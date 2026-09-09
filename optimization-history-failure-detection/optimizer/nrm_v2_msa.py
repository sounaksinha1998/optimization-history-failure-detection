"""NRO-NRMv2-MSA research optimizer with named mode factories.

Extends V5B with multi-scale attention (MSA) denominators. The numerator M^(0)
and nested memories are the same as in ``optimizer.nrm_v2``; only the preconditioner
and per-scale second moments differ.

Equations (per parameter tensor)
-------------------------------

**Shared with V5B** — nested memories G_t, B_t, A_t, long-term L^(j)_t, and
direction M^(0) (see ``optimizer.nrm_v2``).

**Per-timescale second moments** (j = 1 … K):

    V^(j)_t = (1 - λ_j) V^(j)_{t-1} + λ_j g_t²
    V̂^(j)_t = V^(j)_t / (1 - (1 - λ_j)^t)          (bias correction)

**Gradient–memory alignments**:

    M̂^(0) = M^(0) / (‖M^(0)‖ + ε)
    L̂^(j) = L^(j) / (‖L^(j)‖ + ε)
    c_j     = ⟨M̂^(0), L̂^(j)⟩

**MSA attention weights** (mode ``msa``):

    α_j = softmax_j(c_j / τ)

**Blended denominator Ṽ** (mode-dependent):

    adam_d:     Ṽ = V̂_adam     (Adam β₂ second moment, bias-corrected)
    uniform_ms: Ṽ = mean_j V̂^(j)
    msa:        Ṽ = Σ_j α_j V̂^(j)

**Update direction and parameter step**:

    u_t = M^(0) / (√Ṽ + ε)
    θ_{t+1} = θ_t - η u_t

**Memory-derived signals** (logged for diagnostics):

    A     = Σ_j α_j c_j                         (agreement)
    N     = 1 - A                               (novelty)
    D     = Σ_j α_j (c_j - A)²                  (cross-timescale disagreement)
    H^MSA = -Σ_j α_j log α_j                    (MSA entropy)

Modes mirror optimizer-research/nro/optimizer_v5b_msa.py:

    v5b        — original V5B directional sync + D4 denominator
    adam_d     — M^(0) / Adam bias-corrected sqrt(V) + ε
    uniform_ms — M^(0) / sqrt(mean_j V̂^(j)) + ε
    msa        — M^(0) / sqrt(Σ_j α^(j) V̂^(j)) + ε,  α = softmax(c/τ)
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from optax._src import base

from optimizer.core import memory_preconditioned_step
from optimizer.memory import init_long_term_zeros, long_term_accumulation_step
from optimizer.msa import MSA_MODES, MSAMode, msa_direction
from optimizer.timescales import compute_long_term_lambdas, compute_timescale_lambdas, default_timescale_ratio
from optimizer.nrm_v2 import research_nrm_v2

Params = base.Params
Updates = base.Updates
ScalarOrSchedule = base.ScalarOrSchedule


class NRMv2MSAState(NamedTuple):
    """Optimizer state for NRM v2-MSA research skeleton."""

    gamma: Any
    beta: Any
    alpha: Any
    v: Any
    long_term: tuple[Any, ...]
    scale_v: tuple[Any, ...]
    step: jnp.ndarray
    last_alignments: tuple[Any, ...]
    last_alpha: tuple[Any, ...]
    last_agreement: Any
    last_novelty: Any
    last_disagreement: Any
    last_msa_entropy: Any
    last_dominant_scale: Any


def _zero_scalar_trees(params: Params, depth: int) -> tuple[Any, ...]:
    z = jax.tree_util.tree_map(lambda p: jnp.zeros((), dtype=jnp.float32), params)
    return tuple(z for _ in range(depth))


def _init_scale_v_zeros(params: Params, depth: int) -> tuple[Any, ...]:
    return init_long_term_zeros(params, depth)


def research_nrm_v2_msa(
    learning_rate: ScalarOrSchedule,
    *,
    msa_mode: MSAMode = "msa",
    tau: float = 8.0,
    timescale_ratio: float | None = None,
    residual_beta: float = 0.0,
    residual_alpha: float = 0.0,
    long_term_depth: int = 2,
    long_term_timescale_multiplier: float | None = None,
    sync_rho_max: float = 0.25,
    sync_weights: tuple[float, ...] | None = None,
    tau_attention: float = 1.0,
    adam_beta: float = 0.999,
    eps: float = 1e-8,
) -> base.GradientTransformation:
    """Build V5B-MSA or delegate to V5B when ``msa_mode='v5b'``."""
    if msa_mode not in MSA_MODES:
        raise ValueError(f"msa_mode must be one of {MSA_MODES}, got {msa_mode!r}")

    if msa_mode == "v5b":
        return research_nrm_v2(
            learning_rate,
            tau=tau,
            timescale_ratio=timescale_ratio,
            residual_beta=residual_beta,
            residual_alpha=residual_alpha,
            long_term_depth=long_term_depth,
            long_term_timescale_multiplier=long_term_timescale_multiplier,
            sync_rho_max=sync_rho_max,
            sync_weights=sync_weights,
            eps=eps,
            denominator_beta_v=adam_beta,
        )

    if long_term_depth <= 0 and msa_mode in ("uniform_ms", "msa"):
        raise ValueError(f"long_term_depth must be > 0 for msa_mode={msa_mode!r}")

    ratio = timescale_ratio if timescale_ratio is not None else default_timescale_ratio()
    lambdas = compute_timescale_lambdas(tau, ratio)
    lt_lambdas = compute_long_term_lambdas(
        tau, ratio, long_term_depth, long_term_timescale_multiplier
    )

    def init_fn(params: Params) -> NRMv2MSAState:
        zeros = jax.tree_util.tree_map(lambda p: jnp.zeros(p.shape, dtype=jnp.float32), params)
        z_scalar = jax.tree_util.tree_map(lambda p: jnp.zeros((), dtype=jnp.float32), params)
        depth = long_term_depth
        return NRMv2MSAState(
            gamma=zeros,
            beta=zeros,
            alpha=zeros,
            v=zeros,
            long_term=init_long_term_zeros(params, depth),
            scale_v=_init_scale_v_zeros(params, depth),
            step=jnp.array(0, dtype=jnp.int32),
            last_alignments=_zero_scalar_trees(params, depth),
            last_alpha=_zero_scalar_trees(params, depth),
            last_agreement=z_scalar,
            last_novelty=z_scalar,
            last_disagreement=z_scalar,
            last_msa_entropy=z_scalar,
            last_dominant_scale=z_scalar,
        )

    def update_fn(
        grads: Updates, state: NRMv2MSAState, params: Params | None = None
    ) -> tuple[Updates, NRMv2MSAState]:
        del params
        step = state.step + 1
        step_f = step.astype(jnp.float32)
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
        )
        new_long_term = long_term_accumulation_step(mem.alpha, state.long_term, lt_lambdas)

        class _LeafResult(NamedTuple):
            direction: jnp.ndarray
            v_levels: tuple[jnp.ndarray, ...]
            adam_v: jnp.ndarray
            alignments: tuple[jnp.ndarray, ...]
            alpha: tuple[jnp.ndarray, ...]
            agreement: jnp.ndarray
            novelty: jnp.ndarray
            disagreement: jnp.ndarray
            msa_entropy: jnp.ndarray
            dominant: jnp.ndarray

        def leaf_fn(
            m0_l: jnp.ndarray,
            grad_l: jnp.ndarray,
            adam_v_l: jnp.ndarray,
            *level_and_v: jnp.ndarray,
        ) -> _LeafResult:
            n = len(lt_lambdas)
            level_leaves = tuple(level_and_v[:n])
            v_leaves = tuple(level_and_v[n : n + n])
            (
                direction,
                new_v_levels,
                new_adam_v,
                alignments,
                alpha,
                signals,
                dominant,
            ) = msa_direction(
                m0_l,
                grad_l,
                level_leaves,
                v_leaves,
                adam_v_l,
                step_f,
                lt_lambdas,
                msa_mode,
                tau_attention,
                eps,
                adam_beta if msa_mode == "adam_d" else adam_beta,
            )
            return _LeafResult(
                direction=direction,
                v_levels=new_v_levels,
                adam_v=new_adam_v,
                alignments=alignments,
                alpha=alpha,
                agreement=signals.agreement,
                novelty=signals.novelty,
                disagreement=signals.disagreement,
                msa_entropy=signals.msa_entropy,
                dominant=dominant,
            )

        if new_long_term and state.scale_v:
            collected = jax.tree_util.tree_map(
                leaf_fn,
                mem.combined_direction,
                grads,
                state.v,
                *new_long_term,
                *state.scale_v,
                is_leaf=lambda x: isinstance(x, jnp.ndarray),
            )
        else:

            def leaf_no_mem(m0_l, grad_l, adam_v_l):
                direction, new_v_levels, new_adam_v, alignments, alpha, signals, dominant = msa_direction(
                    m0_l,
                    grad_l,
                    (),
                    (),
                    adam_v_l,
                    step_f,
                    lt_lambdas,
                    msa_mode,
                    tau_attention,
                    eps,
                    adam_beta,
                )
                return _LeafResult(
                    direction=direction,
                    v_levels=new_v_levels,
                    adam_v=new_adam_v,
                    alignments=alignments,
                    alpha=alpha,
                    agreement=signals.agreement,
                    novelty=signals.novelty,
                    disagreement=signals.disagreement,
                    msa_entropy=signals.msa_entropy,
                    dominant=dominant,
                )

            collected = jax.tree_util.tree_map(
                leaf_no_mem,
                mem.combined_direction,
                grads,
                state.v,
                is_leaf=lambda x: isinstance(x, jnp.ndarray),
            )

        def _map_leaf(fn):
            return jax.tree_util.tree_map(
                fn, collected, is_leaf=lambda x: isinstance(x, _LeafResult)
            )

        direction = _map_leaf(lambda leaf: leaf.direction)
        new_scale_v = tuple(
            _map_leaf(lambda leaf, k=k: leaf.v_levels[k]) for k in range(len(lt_lambdas))
        )
        new_adam_v = _map_leaf(lambda leaf: leaf.adam_v)

        depth = len(lt_lambdas)
        last_alignments = tuple(
            _map_leaf(lambda leaf, k=k: leaf.alignments[k]) for k in range(depth)
        ) if depth > 0 else ()
        last_alpha = tuple(
            _map_leaf(lambda leaf, k=k: leaf.alpha[k]) for k in range(depth)
        ) if depth > 0 else ()

        lr = (
            learning_rate(step)
            if callable(learning_rate)
            else jnp.asarray(learning_rate, dtype=jnp.float32)
        )
        updates = jax.tree_util.tree_map(lambda d: -lr * d, direction)

        new_state = NRMv2MSAState(
            gamma=mem.gamma,
            beta=mem.beta,
            alpha=mem.alpha,
            v=new_adam_v,
            long_term=new_long_term,
            scale_v=new_scale_v,
            step=step,
            last_alignments=last_alignments,
            last_alpha=last_alpha,
            last_agreement=_map_leaf(lambda leaf: leaf.agreement),
            last_novelty=_map_leaf(lambda leaf: leaf.novelty),
            last_disagreement=_map_leaf(lambda leaf: leaf.disagreement),
            last_msa_entropy=_map_leaf(lambda leaf: leaf.msa_entropy),
            last_dominant_scale=_map_leaf(lambda leaf: leaf.dominant),
        )
        return updates, new_state

    return base.GradientTransformation(init_fn, update_fn)


# Named factories (match optimizer-research/nro/optimizer_v5b_msa.py)

def nro_nrm_v2(*args, **kwargs):
    """Original V5B directional sync."""
    return research_nrm_v2_msa(*args, msa_mode="v5b", **kwargs)


def nro_nrm_v2_adam_d(*args, **kwargs):
    """V5B numerator + Adam denominator (no directional sync)."""
    kwargs.setdefault("sync_rho_max", 0.0)
    return research_nrm_v2_msa(*args, msa_mode="adam_d", **kwargs)


def nro_nrm_v2_uniform_ms(*args, **kwargs):
    """V5B numerator + uniform multi-scale denominator."""
    kwargs.setdefault("sync_rho_max", 0.0)
    return research_nrm_v2_msa(*args, msa_mode="uniform_ms", **kwargs)


def nro_nrm_v2_msa_attention(*args, **kwargs):
    """V5B numerator + attention-selected multi-scale denominator."""
    kwargs.setdefault("sync_rho_max", 0.0)
    return research_nrm_v2_msa(*args, msa_mode="msa", **kwargs)


nro_nrm_v2_msa = research_nrm_v2_msa
