from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, TypeAlias

from .state import GraphState


NodeResult: TypeAlias = dict[str, Any]
Node: TypeAlias = Callable[[GraphState], NodeResult | Awaitable[NodeResult]]

NODE_WRITES: dict[str, frozenset[str]] = {
    "startup_discovery": frozenset(
        {"candidates", "current_idx", "selected_startup", "references"}
    ),
    "customer_profile": frozenset({"profile", "references"}),
    "technical_analysis": frozenset({"technical_analysis", "references"}),
    "business_analysis": frozenset({"business_analysis", "references"}),
    "market_analysis": frozenset({"market_analysis", "references"}),
    "investment_decision": frozenset(
        {
            "scores",
            "investment_score",
            "decision",
            "decision_reason",
            "evaluations",
        }
    ),
    "next_candidate": frozenset({"current_idx", "selected_startup"}),
    "report_generation": frozenset({"report"}),
}


def _validate_update(node_name: str, result: object) -> NodeResult:
    if not isinstance(result, dict):
        raise TypeError(f"{node_name} must return dict, got {type(result).__name__}")
    unexpected = result.keys() - NODE_WRITES[node_name]
    if unexpected:
        keys = ", ".join(sorted(unexpected))
        raise ValueError(f"{node_name} cannot write state keys: {keys}")
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
}


def _node_context(node_name: str, state: GraphState) -> str:
    desc = NODE_DESCRIPTIONS.get(node_name, node_name)
    startup = state.get("selected_startup")
    target = f" [대상: {startup.get('name')}]" if isinstance(startup, dict) and startup.get("name") else ""
    return f"{desc}{target}"


def guard_node(node_name: str, node: Node) -> Node:
    """Reject writes outside a node's ownership contract."""
    # ponytail: stdout printing for CLI progress. upgrade path: logging/callback handler
    if inspect.iscoroutinefunction(node):

        @wraps(node)
        async def async_guarded(state: GraphState) -> NodeResult:
            ctx = _node_context(node_name, state)
            print(f"▶ {ctx} 진행 중...")
            result = await node(state)
            validated = _validate_update(node_name, result)
            print(f"✔ {ctx} 완료")
            return validated

        return async_guarded

    @wraps(node)
    def guarded(state: GraphState) -> NodeResult:
        ctx = _node_context(node_name, state)
        print(f"▶ {ctx} 진행 중...")
        result = node(state)
        if inspect.isawaitable(result):
            raise TypeError(
                f"{node_name} returned an awaitable from a synchronous callable; "
                "declare the node with 'async def'"
            )
        validated = _validate_update(node_name, result)
        print(f"✔ {ctx} 완료")
        return validated

    return guarded
