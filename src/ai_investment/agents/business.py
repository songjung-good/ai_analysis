import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..contracts import NodeResult
from ..models import Evidence
from ..state import GraphState
from ..tools import tools_for
from .business_analysis import analyze_business, cited_source_ids
from .business_models import BusinessAnalysis
from .business_search import build_search_plan, collect_evidence


TOOLS = tools_for("business_analysis")


def run(
    state: GraphState,
    *,
    search: Callable[..., list[Evidence]] | None = None,
    model: Any = None,
) -> NodeResult:
    """Assess deployment stage, customer impact, and regulatory risk."""
    plan = build_search_plan(state)
    if search is None:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)
        if not os.getenv("TAVILY_API_KEY", "").strip():
            raise ValueError("Missing environment variables: TAVILY_API_KEY")
        search = TOOLS[0]
    evidence = collect_evidence(plan, search=search)
    if evidence:
        result = analyze_business(state, evidence, plan.information_gaps, model=model)
    else:
        result = BusinessAnalysis(
            commercialization_stage="unknown",
            stage_reason="검색에서 분석에 사용할 근거를 얻지 못했다.",
            stage_source_ids=[],
            customer_cases=[],
            deployment_costs=[],
            deployment_effects=[],
            regulatory_risks=[],
            evidence_assessment=[],
            information_gaps=[
                *plan.information_gaps,
                "검색 근거 부족: 고객 운영·비용·효과·안전 자료 확인 필요",
            ],
            summary="검색 근거가 없어 현장 도입·사업성을 판단할 수 없다.",
        )
    cited = cited_source_ids(result)
    return {
        "business_analysis": result.model_dump(),
        "references": [item for item in evidence if item.source_id in cited],
    }
