from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from .rag import search_market_docs, search_tech_docs
from .web import web_search


AgentName = Literal[
    "startup_discovery",
    "customer_profile",
    "technical_analysis",
    "business_analysis",
    "market_analysis",
    "investment_decision",
    "report_generation",
]


_TOOLS: dict[AgentName, tuple[Callable[..., object], ...]] = {
    "startup_discovery": (web_search,),
    "customer_profile": (),
    "technical_analysis": (search_tech_docs, web_search),
    "business_analysis": (web_search,),
    "market_analysis": (search_market_docs, web_search),
    "investment_decision": (),
    "report_generation": (),
}


def tools_for(agent: AgentName) -> tuple[Callable[..., object], ...]:
    """Return only the external tools authorized for one Agent."""
    return _TOOLS[agent]
