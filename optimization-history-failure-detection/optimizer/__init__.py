"""Multi-timescale memory research optimizers (NRM v2 / MSA skeleton)."""

from optimizer import associative_memory
from optimizer.msa import MSA_MODES, MSAMode, MemorySignals, compute_memory_signals
from optimizer.nrm_v2 import NRMv2State, research_nrm_v2, research_nrm_v2_observe
from optimizer.nrm_v2_msa import (
    NRMv2MSAState,
    nro_nrm_v2,
    nro_nrm_v2_adam_d,
    nro_nrm_v2_msa,
    nro_nrm_v2_msa_attention,
    nro_nrm_v2_uniform_ms,
    research_nrm_v2_msa,
)

# Deprecated aliases (checkpoint / copied-baseline compatibility)
from optimizer.v5b import V5BState, research_v5b, research_v5b_observe
from optimizer.v5b_msa import (
    V5BMSAState,
    nro_v5b,
    nro_v5b_adam_d,
    nro_v5b_msa,
    nro_v5b_msa_attention,
    nro_v5b_uniform_ms,
    research_v5b_msa,
)

__all__ = [
    "MSA_MODES",
    "MSAMode",
    "MemorySignals",
    "NRMv2MSAState",
    "NRMv2State",
    "V5BMSAState",
    "V5BState",
    "associative_memory",
    "compute_memory_signals",
    "nro_nrm_v2",
    "nro_nrm_v2_adam_d",
    "nro_nrm_v2_msa",
    "nro_nrm_v2_msa_attention",
    "nro_nrm_v2_uniform_ms",
    "nro_v5b",
    "nro_v5b_adam_d",
    "nro_v5b_msa",
    "nro_v5b_msa_attention",
    "nro_v5b_uniform_ms",
    "research_nrm_v2",
    "research_nrm_v2_msa",
    "research_nrm_v2_observe",
    "research_v5b",
    "research_v5b_msa",
    "research_v5b_observe",
]
