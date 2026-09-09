"""Smoke tests for optimizer skeleton."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from optimizer import (
    nro_nrm_v2,
    nro_nrm_v2_adam_d,
    nro_nrm_v2_msa_attention,
    nro_nrm_v2_uniform_ms,
    research_nrm_v2_observe,
)


def _run_one_step(factory):
    params = {"w": jnp.zeros((3, 3), dtype=jnp.float32)}
    grads = {"w": jnp.ones((3, 3), dtype=jnp.float32)}
    tx = factory()
    state = tx.init(params)
    updates, state = tx.update(grads, state, params)
    assert jax.tree_util.tree_leaves(updates)
    return updates, state


def test_research_nrm_v2_observe():
    _run_one_step(lambda: research_nrm_v2_observe(1e-3, long_term_depth=2))


def test_nro_nrm_v2_modes():
    for factory in (nro_nrm_v2, nro_nrm_v2_adam_d, nro_nrm_v2_uniform_ms, nro_nrm_v2_msa_attention):
        _run_one_step(lambda f=factory: f(1e-3, long_term_depth=2, tau_attention=1.0))
