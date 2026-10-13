from __future__ import annotations

import math
from numbers import Real
from typing import Mapping

from pydantic import BaseModel, ConfigDict

from .state import Decision


# 점수가 확인된 항목의 가중치 합이 이 값 이상이면 재정규화한 공식 점수를 낸다.
MIN_COVERAGE = 0.6

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
    coverage: float = 1.0


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
    weighted_score, coverage = provisional_score(normalized)
    if weighted_score is None or coverage < MIN_COVERAGE - 1e-9:
        # 근거가 너무 적으면 미확인 항목을 추정하지 않고 점수를 내지 않는다.
        return ScoreResult(
            scores=normalized, weighted_score=None, decision="hold",
            missing_fields=missing, coverage=coverage,
        )
    # 확인된 항목만 가중치를 재정규화한다(미확인 항목을 1점·중립점으로 채우지 않음).
    decision: Decision = "hold" if force_hold else score_band(weighted_score)
    return ScoreResult(
        scores=normalized, weighted_score=weighted_score, decision=decision,
        missing_fields=missing, coverage=coverage,
    )


def provisional_score(scores: Mapping[str, float]) -> tuple[float | None, float]:
    """Reweight only the scored items; missing items are never imputed.

    Returns (score, scored weight share). calculate_score uses it as the official score
    when the share reaches MIN_COVERAGE; below that it is shown for reference only.
    """
    known = {name: float(value) for name, value in scores.items() if name in WEIGHTS}
    coverage = math.fsum(WEIGHTS[name] for name in known)
    if not known:
        return None, 0.0
    # 재정규화 나눗셈의 부동소수 오차(예: 2.9999999999999996)가 판정 경계를 넘지 않게 정리한다.
    score = math.fsum(value * WEIGHTS[name] for name, value in known.items()) / coverage
    return round(score, 9), coverage


def score_band(score: float) -> Decision:
    """Decision band for a score alone, before missing-item and risk holds."""
    return "invest" if score >= 4.0 else "conditional" if score >= 3.0 else "hold"
