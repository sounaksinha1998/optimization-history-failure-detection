# Optimizer skeleton

Minimal V5B / MSA research optimizers derived from `optimizer-research/nro/`.

## Modes

| Factory | `msa_mode` | Description |
|---------|------------|-------------|
| `research_nrm_v2_observe` | — | Phase 1: track memory, Adam update only |
| `nro_nrm_v2` | `v5b` | V5B directional sync + D4 denominator |
| `nro_nrm_v2_adam_d` | `adam_d` | M^(0) / Adam denominator |
| `nro_nrm_v2_uniform_ms` | `uniform_ms` | M^(0) / uniform multi-scale Ṽ |
| `nro_nrm_v2_msa_attention` | `msa` | M^(0) / attention-weighted Ṽ |

Generic entry point: `research_nrm_v2_msa(msa_mode=...)`.

## Quick start

```python
import optax
from optimizer import nro_nrm_v2_msa_attention, research_nrm_v2_observe

params = {"w": jnp.zeros((4, 4))}
grads = {"w": jnp.ones((4, 4))}

# Phase 1 — memory observation only
tx = research_nrm_v2_observe(learning_rate=1e-3, long_term_depth=2)
state = tx.init(params)
updates, state = tx.update(grads, state, params)

# MSA training path
tx = nro_nrm_v2_msa_attention(learning_rate=1e-3, tau_attention=1.0, long_term_depth=2)
state = tx.init(params)
updates, state = tx.update(grads, state, params)
```

## Layout

```text
optimizer/
├── core.py        # nested memory + M^(0)
├── memory.py      # long-term L^(j) accumulation
├── msa.py         # alignments, attention, Phase-3 signals
├── timescales.py  # τ ↔ λ helpers
├── v5b.py         # V5B + observe mode
└── v5b_msa.py     # named MSA factories
```

See `.cursor/plans/multi-scale-memory-verification.plan.md` for the full research protocol.
