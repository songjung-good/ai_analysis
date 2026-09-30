from __future__ import annotations

import math
from numbers import Real
from typing import Mapping

from pydantic import BaseModel, ConfigDict

from .state import Decision


WEIGHTS = {
    "team": 0.30,
    "market": 0.25,
    "technology": 0.15,
    "competition": 0.10,
    "traction": 0.10,
    "investment_terms": 0.10,
}


class ScoreResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    scores: dict[str, float]
    weighted_score: float | None
    decision: Decision
    missing_fields: list[str]


def calculate_score(
    scores: Mapping[str, float], *, force_hold: bool = False
) -> ScoreResult:
    """Validate scores and calculate the deterministic investment decision."""
    unknown = set(scores) - set(WEIGHTS)
    if unknown:
        raise ValueError(f"unknown score fields: {', '.join(sorted(unknown))}")

    if any(isinstance(value, bool) or not isinstance(value, Real) for value in scores.values()):
        raise ValueError("scores must be real numbers, not booleans or strings")
    normalized = {name: float(value) for name, value in scores.items()}
    invalid = [
        name
        for name, value in normalized.items()
        if not math.isfinite(value) or not 1 <= value <= 5
    ]
    if invalid:
        raise ValueError(f"scores must be between 1 and 5: {', '.join(sorted(invalid))}")
    if not math.isclose(sum(WEIGHTS.values()), 1.0):
        raise RuntimeError("score weights must sum to 1.0")

    missing = sorted(set(WEIGHTS) - set(normalized))
    if missing:
        return ScoreResult(
            scores=normalized,
            weighted_score=None,
            decision="hold",
            missing_fields=missing,
        )

    weighted_score = math.fsum(normalized[name] * weight for name, weight in WEIGHTS.items())
    if force_hold:
        decision: Decision = "hold"
    elif weighted_score >= 4.0:
        decision = "invest"
    elif weighted_score >= 3.0:
        decision = "conditional"
    else:
        decision = "hold"
    return ScoreResult(
        scores=normalized,
        weighted_score=weighted_score,
        decision=decision,
        missing_fields=[],
    )
