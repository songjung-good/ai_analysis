from __future__ import annotations

from typing import Literal

from .state import GraphState


Route = Literal["next_candidate", "report_generation"]


def select_next_candidate(state: GraphState) -> dict[str, object]:
    next_idx = state["current_idx"] + 1
    candidates = state["candidates"]
    if next_idx >= len(candidates):
        raise IndexError("no candidate remains")
    return {
        "current_idx": next_idx,
        "selected_startup": candidates[next_idx],
    }


def route_after_decision(state: GraphState) -> Route:
    decision = state.get("decision")
    if decision == "투자":
        return "report_generation"
    if decision != "보류":
        raise ValueError("decision must be '투자' or '보류'")

    next_idx = state["current_idx"] + 1
    has_candidate = next_idx < len(state["candidates"])
    within_limit = next_idx < state["max_iterations"]
    return "next_candidate" if has_candidate and within_limit else "report_generation"
