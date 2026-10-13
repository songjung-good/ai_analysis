import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..contracts import NodeResult
from ..models import Evidence
from ..state import GraphState, ViewResult
from ..tools import tools_for
from .business_analysis import analyze_business, cited_source_ids
from .business_models import BusinessAnalysis, BusinessView
from .business_search import build_search_plan, collect_evidence


TOOLS = tools_for("business_analysis")


VIEW_REQUIREMENTS = {
    "stakeholder": {
        "customer_buyer": "고객・구매자 역할 확인 필요",
        "user": "실제 사용자 역할 확인 필요",
        "supplier": "공급사 역할 확인 필요",
        "conflicts": "이해 충돌 또는 그 검토 근거 확인 필요",
        "cost_burden": "비용 부담 확인 필요",
    },
    "domain": {
        "application": "적용 현장・업무 확인 필요",
        "deployment": "도입 사례・PoC・운영 단계 확인 필요",
        "constraints": "운영 제약・도입 조건 확인 필요",
    },
}


def to_view_result(result: BusinessAnalysis, *, view: BusinessView) -> ViewResult:
    """Expose cited findings and explicit view coverage without input-only gaps."""
    if view not in VIEW_REQUIREMENTS:
        raise ValueError("view must be 'stakeholder' or 'domain'")
    requirements = VIEW_REQUIREMENTS[view]
    topics = {item.topic for item in result.view_findings}
    if topics - requirements.keys():
        raise ValueError(f"{view} result contains another view's topics")
    claims = [
        {"text": item.text, "source_ids": list(dict.fromkeys(item.source_ids))}
        for item in result.view_findings
    ]
    missing = list(result.view_missing_items)
    missing.extend(message for topic, message in requirements.items() if topic not in topics)
    return {
        "view": view, "summary": result.summary, "claims": claims,
        "missing_items": list(dict.fromkeys(missing)),
        "source_ids": sorted({sid for claim in claims for sid in claim["source_ids"]}),
        "status": "done",
    }


def run(
    state: GraphState,
    *,
    view: BusinessView | None = None,
    search: Callable[..., list[Evidence]] | None = None,
    model: Any = None,
) -> NodeResult:
    """Run legacy business analysis or research one designated business view."""
    plan = build_search_plan(state, view=view)
    try:
        if search is None:
            from dotenv import load_dotenv

            load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)
            if not os.getenv("TAVILY_API_KEY", "").strip():
                raise ValueError("Missing environment variables: TAVILY_API_KEY")
            search = TOOLS[0]
        evidence = collect_evidence(plan, search=search)
        if evidence:
            result = analyze_business(state, evidence, plan.information_gaps, model=model, view=view)
        else:
            result = BusinessAnalysis(
                commercialization_stage="unknown",
                stage_reason="검색에서 분석에 사용할 근거를 얻지 못했다.",
                stage_source_ids=[], customer_cases=[], deployment_costs=[],
                deployment_effects=[], regulatory_risks=[], evidence_assessment=[],
                information_gaps=[*plan.information_gaps, "검색 근거 부족: 추가 자료 확인 필요"],
                summary="검색 근거가 없어 해당 관점을 판단할 수 없다.",
            )
        if view is not None:
            analysis = to_view_result(result, view=view)
            return {
                "analyses": {view: analysis},
                "references": [item for item in evidence if item.source_id in analysis["source_ids"]],
                "node_status": {view: "done"}, "node_errors": {view: None},
            }
    except Exception as exc:
        if view is None:
            raise
        return {
            "analyses": {view: {
                "view": view, "summary": "분석 실행 실패", "claims": [], "source_ids": [],
                "missing_items": list(VIEW_REQUIREMENTS[view].values()), "status": "failed",
            }},
            "references": [], "node_status": {view: "failed"},
            "node_errors": {view: f"{type(exc).__name__}: {exc}"},
        }
    cited = cited_source_ids(result)
    return {
        "business_analysis": result.model_dump(),
        "references": [item for item in evidence if item.source_id in cited],
    }
