"""Deprecated: use `optimizer.nrm_v2`."""

from optimizer.nrm_v2 import (
    NRMv2State,
    apply_directional_sync,
    research_nrm_v2,
    research_nrm_v2_observe,
    resolve_sync_weights,
)

V5BState = NRMv2State
research_v5b = research_nrm_v2
research_v5b_observe = research_nrm_v2_observe

__all__ = [
    "V5BState",
    "NRMv2State",
    "apply_directional_sync",
    "research_v5b",
    "research_v5b_observe",
    "research_nrm_v2",
    "research_nrm_v2_observe",
    "resolve_sync_weights",
]
