"""Timescale parameterization for nested memories."""

from __future__ import annotations

import math


def default_timescale_ratio() -> float:
    """Return φ = (1 + √5) / 2."""
    return (1.0 + math.sqrt(5.0)) / 2.0


def lambda_from_tau(tau: float) -> float:
    """Convert timescale τ to EMA coefficient λ = 1 - e^{-1/τ}."""
    if tau <= 0.0:
        raise ValueError(f"tau must be positive, got {tau}")
    return 1.0 - math.exp(-1.0 / tau)


def compute_timescale_lambdas(tau: float, timescale_ratio: float) -> dict[str, float]:
    """Compute τ and λ for fast / medium / slow nested memories."""
    if tau <= 0.0:
        raise ValueError(f"tau must be positive, got {tau}")
    if timescale_ratio <= 0.0:
        raise ValueError(f"timescale_ratio must be positive, got {timescale_ratio}")

    r = timescale_ratio
    tau_gamma = tau
    tau_beta = r * tau_gamma
    tau_alpha = r * r * tau_gamma

    return {
        "tau_gamma": tau_gamma,
        "tau_beta": tau_beta,
        "tau_alpha": tau_alpha,
        "lambda_gamma": lambda_from_tau(tau_gamma),
        "lambda_beta": lambda_from_tau(tau_beta),
        "lambda_alpha": lambda_from_tau(tau_alpha),
        "timescale_ratio": r,
    }


def compute_long_term_lambdas(
    tau: float,
    timescale_ratio: float,
    long_term_depth: int,
    long_term_timescale_multiplier: float | None = None,
) -> tuple[float, ...]:
    """Resolve λ_j for hierarchical long-term memories with τ_{j+1} = q τ_j."""
    if long_term_depth < 0:
        raise ValueError(f"long_term_depth must be >= 0, got {long_term_depth}")
    if long_term_depth == 0:
        return ()

    q = (
        long_term_timescale_multiplier
        if long_term_timescale_multiplier is not None
        else timescale_ratio
    )
    base = compute_timescale_lambdas(tau, timescale_ratio)
    tau_lt = base["tau_alpha"] * q
    lambdas: list[float] = []
    for _ in range(long_term_depth):
        lambdas.append(lambda_from_tau(tau_lt))
        tau_lt *= q
    return tuple(lambdas)
