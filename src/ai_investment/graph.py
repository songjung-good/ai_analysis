from __future__ import annotations

from dataclasses import dataclass

from .contracts import Node, guard_node
from .control import route_after_decision, route_after_discovery, select_next_candidate
from .state import GraphState


@dataclass(frozen=True)
class AgentNodes:
    startup_discovery: Node
    customer_profile: Node
    technical_analysis: Node
    business_analysis: Node
    market_analysis: Node
    investment_decision: Node
    report_generation: Node


def build_graph(nodes: AgentNodes):
    """Build the design document's main graph with conflict-checked nodes."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc

    builder = StateGraph(GraphState)
    builder.add_node(
        "startup_discovery", guard_node("startup_discovery", nodes.startup_discovery)
    )
    builder.add_node(
        "customer_profile", guard_node("customer_profile", nodes.customer_profile)
    )
    builder.add_node(
        "technical_analysis",
        guard_node("technical_analysis", nodes.technical_analysis),
    )
    builder.add_node(
        "business_analysis", guard_node("business_analysis", nodes.business_analysis)
    )
    builder.add_node(
        "market_analysis", guard_node("market_analysis", nodes.market_analysis)
    )
    builder.add_node(
        "investment_decision",
        guard_node("investment_decision", nodes.investment_decision),
    )
    builder.add_node(
        "next_candidate", guard_node("next_candidate", select_next_candidate)
    )
    builder.add_node(
        "report_generation",
        guard_node("report_generation", nodes.report_generation),
    )

    builder.add_edge(START, "startup_discovery")
    builder.add_conditional_edges(
        "startup_discovery",
        route_after_discovery,
        {"customer_profile": "customer_profile", "report_generation": "report_generation"},
    )
    builder.add_edge("customer_profile", "technical_analysis")
    builder.add_edge("customer_profile", "business_analysis")
    builder.add_edge("customer_profile", "market_analysis")
    builder.add_edge(
        ["technical_analysis", "business_analysis", "market_analysis"],
        "investment_decision",
    )
    builder.add_conditional_edges(
        "investment_decision",
        route_after_decision,
        {
            "next_candidate": "next_candidate",
            "report_generation": "report_generation",
        },
    )
    builder.add_edge("next_candidate", "customer_profile")
    builder.add_edge("report_generation", END)
    return builder.compile()
