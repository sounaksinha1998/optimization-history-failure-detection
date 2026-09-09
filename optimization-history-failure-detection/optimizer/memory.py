"""Multi-timescale long-term memory helpers.

Equations (per parameter tensor; applied elementwise across the param tree)
--------------------------------------------------------------------------

**Hierarchical long-term memories** L^(1), …, L^(K) are updated from the
slow nested signal A_t (source) with EMA coefficients λ_1 > λ_2 > … > λ_K:

    L^(1)_t = (1 - λ_1) L^(1)_{t-1} + λ_1 A_t
    L^(j)_t = (1 - λ_j) L^(j)_{t-1} + λ_j L^(j-1)_t,   j = 2, …, K

Each level feeds the next: the output of level j − 1 is the input to level j.
λ_j values are derived from timescale τ via λ(τ) = 1 − exp(−1/τ) with
progressively slower timescales (see ``optimizer.timescales``).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def init_long_term_zeros(params: Any, depth: int) -> tuple[Any, ...]:
    """Initialize L^(1)...L^(K) to zeros matching params."""
    if depth <= 0:
        return ()
    zeros = jax.tree_util.tree_map(lambda p: jnp.zeros(p.shape, dtype=jnp.float32), params)
    return tuple(zeros for _ in range(depth))


def long_term_accumulation_step(
    source: Any,
    long_term: tuple[Any, ...],
    lambda_long_term: tuple[float, ...],
) -> tuple[Any, ...]:
    """Advance L^(1)...L^(K) from source signal A_t and previous states."""
    if not lambda_long_term:
        return ()

    new_levels: list[Any] = []
    current = source
    for level_idx, lam in enumerate(lambda_long_term):
        prev = long_term[level_idx] if level_idx < len(long_term) else None
        if prev is None:
            prev = jax.tree_util.tree_map(jnp.zeros_like, source)
        new_level = jax.tree_util.tree_map(
            lambda src, mem: (1.0 - lam) * mem + lam * src,
            current,
            prev,
        )
        new_levels.append(new_level)
        current = new_level
    return tuple(new_levels)
