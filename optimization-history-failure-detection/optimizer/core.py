"""Minimal NRO-V3 memory step used by V5B / MSA research optimizers."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


class MemoryStep(NamedTuple):
    """Result of one nested-memory preconditioning step."""

    gamma: Any
    beta: Any
    alpha: Any
    v: Any
    combined_direction: Any
    denominator: Any
    direction: Any


def nested_memory_step(
    grads: Any,
    gamma: Any,
    beta: Any,
    alpha: Any,
    lambda_gamma: float,
    lambda_beta: float,
    lambda_alpha: float,
    *,
    warm_start: bool = False,
) -> tuple[Any, Any, Any]:
    """Update G_t, B_t, A_t nested memories."""

    def _ema(prev, g, lam):
        if warm_start:
            return g
        return (1.0 - lam) * prev + lam * g

    new_gamma = jax.tree_util.tree_map(
        lambda p, g: _ema(p, g, lambda_gamma), gamma, grads
    )
    new_beta = jax.tree_util.tree_map(
        lambda p, g: _ema(p, g, lambda_beta), beta, new_gamma
    )
    new_alpha = jax.tree_util.tree_map(
        lambda p, g: _ema(p, g, lambda_alpha), alpha, new_beta
    )
    return new_gamma, new_beta, new_alpha


def compute_normalized_residual_direction(
    gamma: Any,
    beta: Any,
    alpha: Any,
    residual_beta: float,
    residual_alpha: float,
    eps: float,
) -> Any:
    """M^(0) = G ⊙ (1 + c1 R_GB + c2 R_BA) with normalized residuals."""

    def _rgb(g, b):
        return (g - b) / (jnp.abs(g) + jnp.abs(b) + eps)

    def _rba(b, a):
        return (b - a) / (jnp.abs(b) + jnp.abs(a) + eps)

    def _combine(g, b, a):
        return g * (1.0 + residual_beta * _rgb(g, b) + residual_alpha * _rba(b, a))

    return jax.tree_util.tree_map(_combine, gamma, beta, alpha)


def update_second_moment(v: Any, grads: Any, beta_v: float) -> Any:
    """Adam-like second moment V_t = β V_{t-1} + (1-β) g^2."""
    return jax.tree_util.tree_map(
        lambda vv, g: beta_v * vv + (1.0 - beta_v) * jnp.square(g),
        v,
        grads,
    )


def adam_denominator(v: Any, step: jnp.ndarray, beta_v: float, eps: float) -> Any:
    """Bias-corrected sqrt(V) + ε."""
    step_f = jnp.asarray(step, dtype=jnp.float32)
    correction = 1.0 - jnp.power(jnp.asarray(beta_v, dtype=jnp.float32), step_f)
    return jax.tree_util.tree_map(
        lambda vv: jnp.sqrt(vv / jnp.maximum(correction, 1e-12)) + eps,
        v,
    )


def d4_denominator(gamma: Any, beta: Any, alpha: Any, eps: float) -> Any:
    """NRO D4 denominator: sqrt(|A|) + ε."""
    return jax.tree_util.tree_map(lambda a: jnp.sqrt(jnp.abs(a)) + eps, alpha)


def memory_preconditioned_step(
    grads: Any,
    gamma: Any,
    beta: Any,
    alpha: Any,
    v: Any,
    lambda_gamma: float,
    lambda_beta: float,
    lambda_alpha: float,
    eps: float,
    *,
    warm_start: bool = False,
    residual_beta: float = 0.0,
    residual_alpha: float = 0.0,
    denominator: str = "D4",
    denominator_beta_v: float = 0.999,
) -> MemoryStep:
    """One NRO-V3-style step returning M^(0) and denominator."""
    new_gamma, new_beta, new_alpha = nested_memory_step(
        grads,
        gamma,
        beta,
        alpha,
        lambda_gamma,
        lambda_beta,
        lambda_alpha,
        warm_start=warm_start,
    )
    combined = compute_normalized_residual_direction(
        new_gamma,
        new_beta,
        new_alpha,
        residual_beta,
        residual_alpha,
        eps,
    )

    if denominator.upper() == "ADAM":
        new_v = update_second_moment(v, grads, denominator_beta_v)
        denom = adam_denominator(new_v, jnp.asarray(1, dtype=jnp.int32), denominator_beta_v, eps)
    elif denominator.upper() == "D4":
        new_v = v
        denom = d4_denominator(new_gamma, new_beta, new_alpha, eps)
    else:
        raise ValueError(f"Unsupported denominator {denominator!r}; use 'D4' or 'ADAM'")

    direction = jax.tree_util.tree_map(lambda m, d: m / d, combined, denom)
    return MemoryStep(
        gamma=new_gamma,
        beta=new_beta,
        alpha=new_alpha,
        v=new_v,
        combined_direction=combined,
        denominator=denom,
        direction=direction,
    )
