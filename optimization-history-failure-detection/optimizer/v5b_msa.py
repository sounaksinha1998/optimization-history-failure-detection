"""Deprecated: use `optimizer.nrm_v2_msa`."""

from optimizer.nrm_v2_msa import (
    NRMv2MSAState,
    nro_nrm_v2,
    nro_nrm_v2_adam_d,
    nro_nrm_v2_msa_attention,
    nro_nrm_v2_uniform_ms,
    research_nrm_v2_msa,
)

V5BMSAState = NRMv2MSAState
nro_v5b = nro_nrm_v2
nro_v5b_adam_d = nro_nrm_v2_adam_d
nro_v5b_uniform_ms = nro_nrm_v2_uniform_ms
nro_v5b_msa_attention = nro_nrm_v2_msa_attention
nro_v5b_msa = research_nrm_v2_msa
research_v5b_msa = research_nrm_v2_msa

__all__ = [
    "V5BMSAState",
    "NRMv2MSAState",
    "nro_v5b",
    "nro_v5b_adam_d",
    "nro_v5b_msa",
    "nro_v5b_msa_attention",
    "nro_v5b_uniform_ms",
    "nro_nrm_v2",
    "nro_nrm_v2_adam_d",
    "nro_nrm_v2_msa_attention",
    "nro_nrm_v2_uniform_ms",
    "research_v5b_msa",
    "research_nrm_v2_msa",
]
