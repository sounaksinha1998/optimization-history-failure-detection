"""Tests for clinical failure metric helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.common.clinical_failure import aurc_from_curve


def test_aurc_from_curve_uses_numpy_trapezoid_api() -> None:
    curve = pd.DataFrame({"coverage": [0.5, 1.0], "risk": [0.4, 0.2]})
    aurc = aurc_from_curve(curve)
    if hasattr(np, "trapezoid"):
        expected = float(np.trapezoid(curve["risk"], curve["coverage"]))
    else:
        expected = float(np.trapz(curve["risk"], curve["coverage"]))
    assert aurc == pytest.approx(expected)
