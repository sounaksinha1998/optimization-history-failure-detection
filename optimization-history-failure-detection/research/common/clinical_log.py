"""Flush-safe progress logging for clinical experiment runs."""

from __future__ import annotations


def clinical_log(message: str, *, verbose: int, level: int = 1) -> None:
    """Print when ``verbose >= level``. Level 1=phases, 2=epochs/details, 3=fine-grained."""
    if verbose >= level:
        print(message, flush=True)
