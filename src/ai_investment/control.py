from __future__ import annotations

from typing import Literal

from .state import GraphState, candidate_reset_update


Route = Literal["next_candidate", "report_generation"]

MAX_REPORT_FAILURES = 3  # STATE_CONTRACT: 3번째 미달에서 종료


def route_after_discovery(state: GraphState) -> Literal["customer_profile", "report_generation"]:
    return "customer_profile" if state.get("candidates") else "report_generation"


def select_next_candidate(state: GraphState) -> dict[str, object]:
    next_idx = state["current_idx"] + 1
    candidates = state["candidates"]
    if next_idx >= len(candidates):
        raise IndexError("no candidate remains")
    return {
        **candidate_reset_update(),  # 이전 후보 결과 초기화
        "current_idx": next_idx,
        "selected_startup": candidates[next_idx],
    }


def route_after_decision(state: GraphState) -> Route:
    decision = state.get("decision")
    if decision == "invest":
        return "report_generation"
    if decision not in {"conditional", "hold"}:
        raise ValueError("decision must be 'invest', 'conditional', or 'hold'")

    next_idx = state["current_idx"] + 1
    has_candidate = next_idx < len(state["candidates"])
    within_limit = next_idx < state["max_iterations"]
    return "next_candidate" if has_candidate and within_limit else "report_generation"


def route_from_supervisor(state: GraphState):
    # 이번 라운드에 배정된 관점이 있으면 그 목록(병렬 실행), 없으면 복귀 지점으로
    views = list(state.get("next_views") or [])
    return views if views else state.get("resume_after_research", "investment_decision")


def _exhausted(state: GraphState) -> bool:
    verdict = state.get("eval_result") or {}
    return (not verdict.get("passed")) and state.get("report_attempts", 0) >= MAX_REPORT_FAILURES


def route_after_report(state: GraphState) -> str:
    # 통과/한계 보고서에 최종 평가 상태를 반영했으면 종료, 아니면 평가.
    passed = (state.get("eval_result") or {}).get("passed") is True
    return "end" if passed or _exhausted(state) else "evaluator"


def route_after_evaluation(state: GraphState) -> str:
    verdict = state.get("eval_result") or {}
    if verdict.get("passed"):
        return "report_generation"  # 평가받은 본문을 유지하며 최종 통과 상태 출력
    if _exhausted(state):
        return "report_generation"  # 미통과 사유를 반영한 한계 보고서를 한 번 더 출력
    return "supervisor" if verdict.get("action") == "research" else "report_generation"
