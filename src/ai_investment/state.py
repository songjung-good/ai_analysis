from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict


Decision = Literal["투자", "보류"]


class Startup(TypedDict, total=False):
    name: str
    product: str
    funding_stage: str
    team: list[str]


class Reference(TypedDict):
    source_id: str
    title: str
    location: str
    startup_name: str
    agent: str


class Evaluation(TypedDict):
    startup_name: str
    score: float
    decision: Decision
    reason: str


class GraphState(TypedDict, total=False):
    domain: str
    criteria: dict[str, object]
    candidates: list[Startup]
    current_idx: int
    max_iterations: int
    selected_startup: Startup
    profile: dict[str, object]
    technical_analysis: dict[str, object]
    business_analysis: dict[str, object]
    market_analysis: dict[str, object]
    scores: dict[str, float]
    investment_score: float
    decision: Decision
    decision_reason: str
    evaluations: Annotated[list[Evaluation], operator.add]
    references: Annotated[list[Reference], operator.add]
    report: str


def create_initial_state(
    *, domain: str, criteria: dict[str, object], max_iterations: int
) -> GraphState:
    """Create validated graph input without agent-owned output fields."""
    if not domain.strip():
        raise ValueError("domain must not be empty")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    return {
        "domain": domain,
        "criteria": criteria,
        "max_iterations": max_iterations,
        "evaluations": [],
        "references": [],
    }
