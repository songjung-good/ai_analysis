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


def guard_node(node_name: str, node: Node) -> Node:
    """Reject writes outside a node's ownership contract."""
    if inspect.iscoroutinefunction(node):

        @wraps(node)
        async def async_guarded(state: GraphState) -> NodeResult:
            return _validate_update(node_name, await node(state))

        return async_guarded

    @wraps(node)
    def guarded(state: GraphState) -> NodeResult:
        result = node(state)
        if inspect.isawaitable(result):
            raise TypeError(
                f"{node_name} returned an awaitable from a synchronous callable; "
                "declare the node with 'async def'"
            )
        return _validate_update(node_name, result)

    return guarded
