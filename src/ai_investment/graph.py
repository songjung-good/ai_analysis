from __future__ import annotations

from dataclasses import dataclass

from .contracts import Node, guard_node
from .control import (
    route_after_decision,
    route_after_discovery,
    route_after_evaluation,
    route_after_report,
    route_from_supervisor,
    select_next_candidate,
)
from .state import VIEWS, GraphState


@dataclass(frozen=True)
class AgentNodes:
    startup_discovery: Node
    customer_profile: Node
    technical_analysis: Node
    business_analysis: Node
    market_analysis: Node
    investment_decision: Node
    report_generation: Node
    # supervisor를 지정하면 supervisor 그래프, 없으면 기존 고정 fan-out 그래프
    supervisor: Node | None = None
    stakeholder: Node | None = None
    domain: Node | None = None
    evaluator: Node | None = None


def build_graph(nodes: AgentNodes):
    """Build the main graph with conflict-checked nodes."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc

    supervised = nodes.supervisor is not None
    if supervised and None in (nodes.stakeholder, nodes.domain, nodes.evaluator):
        raise ValueError("supervisor graph needs stakeholder, domain and evaluator nodes")

    builder = StateGraph(GraphState)

    def add(name, fn):
        builder.add_node(name, guard_node(name, fn))

    add("startup_discovery", nodes.startup_discovery)
    add("customer_profile", nodes.customer_profile)
    add("investment_decision", nodes.investment_decision)
    add("next_candidate", select_next_candidate)
    add("report_generation", nodes.report_generation)

    builder.add_edge(START, "startup_discovery")
    builder.add_conditional_edges(
        "startup_discovery",
        route_after_discovery,
        {"customer_profile": "customer_profile", "report_generation": "report_generation"},
    )
    builder.add_conditional_edges(
        "investment_decision",
        route_after_decision,
        {"next_candidate": "next_candidate", "report_generation": "report_generation"},
    )
    builder.add_edge("next_candidate", "customer_profile")

    if not supervised:  # 기존 고정 구조 (기존 테스트 호환)
        add("technical_analysis", nodes.technical_analysis)
        add("business_analysis", nodes.business_analysis)
        add("market_analysis", nodes.market_analysis)
        builder.add_edge("customer_profile", "technical_analysis")
        builder.add_edge("customer_profile", "business_analysis")
        builder.add_edge("customer_profile", "market_analysis")
        builder.add_edge(
            ["technical_analysis", "business_analysis", "market_analysis"],
            "investment_decision",
        )
        builder.add_edge("report_generation", END)
        return builder.compile()

    # supervisor 구조: 하위 노드는 supervisor로만 복귀 (개별 엣지: 배정된 관점만 실행되므로)
    workers = {
        "technical": nodes.technical_analysis,
        "market": nodes.market_analysis,
        "stakeholder": nodes.stakeholder,
        "domain": nodes.domain,
    }
    add("supervisor", nodes.supervisor)
    for view in VIEWS:
        add(view, workers[view])
        builder.add_edge(view, "supervisor")
    add("evaluator", nodes.evaluator)

    builder.add_edge("customer_profile", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            **{view: view for view in VIEWS},
            "investment_decision": "investment_decision",
            "report_generation": "report_generation",
        },
    )
    builder.add_conditional_edges(
        "report_generation",
        route_after_report,
        {"evaluator": "evaluator", "end": END},
    )
    builder.add_conditional_edges(
        "evaluator",
        route_after_evaluation,
        {"end": END, "supervisor": "supervisor", "report_generation": "report_generation"},
    )
    return builder.compile()
