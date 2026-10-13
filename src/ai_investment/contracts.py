from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, TypeAlias

from .state import GraphState
from .utils.logger import monitor_node


NodeResult: TypeAlias = dict[str, Any]
Node: TypeAlias = Callable[[GraphState], NodeResult | Awaitable[NodeResult]]

NODE_WRITES: dict[str, frozenset[str]] = {
    "startup_discovery": frozenset(
        {"candidates", "current_idx", "selected_startup", "references"}
    ),
    "customer_profile": frozenset({"profile", "references"}),
    "technical_analysis": frozenset(
        {"technical_analysis", "analyses", "references", "node_status", "node_errors"}
    ),
    "business_analysis": frozenset({"business_analysis", "references"}),
    "market_analysis": frozenset(
        {"market_analysis", "analyses", "references", "node_status", "node_errors"}
    ),
    "supervisor": frozenset(
        {"next_views", "rework_count", "rework_hints", "step_count", "eval_result"}
    ),
    "technical": frozenset(
        {"technical_analysis", "analyses", "references", "node_status", "node_errors"}
    ),
    "market": frozenset(
        {"market_analysis", "analyses", "references", "node_status", "node_errors"}
    ),
    "stakeholder": frozenset({"analyses", "references", "node_status", "node_errors"}),
    "domain": frozenset({"analyses", "references", "node_status", "node_errors"}),
    "investment_decision": frozenset(
        {
            "scores",
            "investment_score",
            "decision",
            "decision_reason",
            "evaluations",
        }
    ),
    "next_candidate": frozenset(
        {
            "current_idx", "selected_startup", "analyses", "node_status", "node_errors",
            "next_views", "rework_count", "rework_hints", "eval_result",
            "report_draft", "report_attempts", "report",
            "profile", "technical_analysis", "business_analysis", "market_analysis",
            "scores", "investment_score", "decision", "decision_reason",
            "resume_after_research",
        }
    ),
    "report_generation": frozenset({"report", "report_draft"}),
    "evaluator": frozenset({"eval_result", "report_attempts", "resume_after_research"}),
}

WORKER_VIEWS = {
    "technical": "technical", "technical_analysis": "technical",
    "market": "market", "market_analysis": "market",
    "stakeholder": "stakeholder", "domain": "domain",
}


def _validate_update(node_name: str, result: object) -> NodeResult:
    if not isinstance(result, dict):
        raise TypeError(f"{node_name} must return dict, got {type(result).__name__}")
    unexpected = result.keys() - NODE_WRITES[node_name]
    if unexpected:
        keys = ", ".join(sorted(unexpected))
        raise ValueError(f"{node_name} cannot write state keys: {keys}")
    view = WORKER_VIEWS.get(node_name)
    if view is not None:
        for field in ("analyses", "node_status", "node_errors"):
            if field not in result:
                continue
            update = result[field]
            if not isinstance(update, dict):
                raise TypeError(f"{node_name}.{field} must be dict")
            if update.keys() - {view}:
                raise ValueError(f"{node_name} can only write {field}[{view!r}]")
        if view in result.get("analyses", {}):
            analysis = result["analyses"][view]
            if not isinstance(analysis, dict) or analysis.get("view") != view:
                raise ValueError(f"{node_name} analysis.view must be {view!r}")
    return result


NODE_DESCRIPTIONS: dict[str, str] = {
    "startup_discovery": "1단계: 투자 후보 스타트업 탐색 및 검증",
    "customer_profile": "2단계: 고객 프로필 및 도메인 분류",
    "technical_analysis": "3단계: 기술 검증 및 특허/논문 분석",
    "business_analysis": "4단계: 현장 도입 및 사업성 분석",
    "market_analysis": "5단계: 시장 규모 및 경쟁 분석",
    "investment_decision": "6단계: 투자 평가 점수 및 최종 판단",
    "next_candidate": "다음 후보 기업 전환",
    "report_generation": "7단계: 투자 보고서 PDF 생성",
    "supervisor": "관점별 근거 충분성 판단 및 작업 배정",
    "technical": "기술 성숙도 분석",
    "market": "시장성 분석",
    "stakeholder": "이해관계자 분석",
    "domain": "도메인 적용 분석",
    "evaluator": "보고서 품질 평가",
}


def guard_node(node_name: str, node: Node) -> Node:
    """Reject writes outside a node's ownership contract."""
    if inspect.iscoroutinefunction(node):

        @wraps(node)
        async def async_guarded(state: GraphState) -> NodeResult:
            with monitor_node(node_name, state, NODE_DESCRIPTIONS.get(node_name, node_name)) as metrics:
                validated = _validate_update(node_name, await node(state))
                _result_metrics(metrics, validated)
                return validated

        return async_guarded

    @wraps(node)
    def guarded(state: GraphState) -> NodeResult:
        with monitor_node(node_name, state, NODE_DESCRIPTIONS.get(node_name, node_name)) as metrics:
            result = node(state)
            if inspect.isawaitable(result):
                raise TypeError(
                    f"{node_name} returned an awaitable from a synchronous callable; "
                    "declare the node with 'async def'"
                )
            validated = _validate_update(node_name, result)
            _result_metrics(metrics, validated)
            return validated

    return guarded


def _result_metrics(metrics, result):
    analyses = result.get("analyses") or {}
    if not isinstance(analyses, dict):  # candidate reset uses LangGraph Overwrite
        analyses = {}
    errors = {view: (result.get("node_errors") or {}).get(view) or "분석 실행 실패"
              for view, value in analyses.items()
              if value.get("status") == "failed"}
    missing = sum(len(value.get("missing_items", [])) for value in analyses.values())
    metrics.update(result_count=len(result.get("references") or []), missing_count=missing)
    if "candidates" in result:
        metrics["candidate_count"] = len(result["candidates"])
    if errors:
        metrics.update(status="failed", error=errors)
    elif missing:
        metrics["status"] = "insufficient"
