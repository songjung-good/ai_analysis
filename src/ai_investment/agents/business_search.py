"""Bounded web research for the deployment and business Agent."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ..models import Evidence
from ..state import GraphState, dedupe_by_source_id
from ..tools.web import web_search
from ..utils.logger import monitor
from .business_models import BusinessView, validate_business_view


MAX_SEARCH_CALLS = 3
MAX_RESULTS_PER_CALL = 5


@dataclass(frozen=True)
class SearchPlan:
    queries: tuple[str, ...]
    information_gaps: tuple[str, ...]
    rework_hints: tuple[str, ...] = ()


def _mapping(value: object, field: str) -> Mapping:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _text(value: object, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    return value.strip()


def _strings(value: object, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{field} must be a list of strings")
    return [item.strip() for item in value if item.strip()]


def build_search_plan(state: GraphState, *, view: BusinessView | None = None) -> SearchPlan:
    """Validate known input fields without mutating the shared State."""
    validate_business_view(view)
    startup = _mapping(state.get("selected_startup"), "selected_startup")
    profile = _mapping(state.get("profile"), "profile")
    name = _text(startup.get("name"), "selected_startup.name")
    if not name:
        raise ValueError("selected_startup.name is required")

    gaps = []
    values = {}
    for field in ("product", "funding_stage"):
        values[field] = _text(startup.get(field), f"selected_startup.{field}")
        if not values[field]:
            gaps.append(f"입력 누락: selected_startup.{field}")
    if not _strings(startup.get("team"), "selected_startup.team"):
        gaps.append("입력 누락: selected_startup.team")
    for field in ("subdomain", "paying_customer", "customer_problem", "info_sufficiency"):
        values[field] = _text(profile.get(field), f"profile.{field}")
        if not values[field]:
            gaps.append(f"입력 누락: profile.{field}")
    if profile.get("missing_info") is None:
        gaps.append("입력 누락: profile.missing_info")
    gaps.extend(_strings(profile.get("missing_info"), "profile.missing_info"))

    # Use company-only queries when product or customer context is incomplete.
    context_complete = all(values[field] for field in (
        "product", "subdomain", "paying_customer", "customer_problem"
    ))
    subject = " ".join((name, values["product"])) if context_complete else name
    customer = f" {values['paying_customer']}" if context_complete else ""
    if view == "stakeholder":
        queries = (
            f"{subject}{customer} 고객 구매자 사용자 공급사 stakeholder buyer user supplier",
            f"{subject} 구매 의사결정 비용 부담 수익 배분 procurement cost responsibility",
            f"{subject} 파트너 작업자 이해 충돌 도입 반대 stakeholder conflict worker impact",
        )
    elif view == "domain":
        queries = (
            f"{subject}{customer} 적용 현장 도입 사례 PoC 실제 운영 deployment pilot operation",
            f"{subject} 설치 통합 운영 조건 현장 제약 deployment integration limitations",
            f"{subject} 현장 실패 안전 규제 인증 deployment failure safety certification",
        )
    else:
        queries = (
            f"{subject}{customer} 고객 도입 유료 계약 시험 운영 customer deployment paid contract pilot",
            f"{subject} 가격 설치 유지보수 생산성 pricing maintenance productivity ROI case study",
            f"{subject} 안전 인증 현장 도입 safety certification deployment",
        )
    if view is not None:
        unrelated = {
            "입력 누락: selected_startup.funding_stage", "입력 누락: selected_startup.team",
            "입력 누락: profile.info_sufficiency", "입력 누락: profile.missing_info",
        }
        gaps = [gap for gap in gaps if gap not in unrelated]
    hints = ()
    if view is not None:
        requests = _mapping(state.get("rework_hints"), "rework_hints")
        hints = tuple(dict.fromkeys(_strings(requests.get(view), f"rework_hints.{view}")))
        if hints:
            focus = " ".join(hints)
            scope = "이해관계자" if view == "stakeholder" else "도메인 적용"
            queries = tuple(
                f"{subject}{customer} {scope} {focus} {angle}"
                for angle in ("고객 사례 원자료", "독립 검증 한계 반대 근거", "공식 자료 조건 위험")
            )
    return SearchPlan(queries, tuple(dict.fromkeys(gaps)), hints)


def collect_evidence(
    plan: SearchPlan, *, search: Callable[..., list[Evidence]] = web_search,
    trace_id: str | None = None, view: BusinessView | None = None, attempt: int = 0,
    target_name: str | None = None,
) -> list[Evidence]:
    """Search once per topic; propagate failures and merge repeated source IDs."""
    if len(plan.queries) != MAX_SEARCH_CALLS:
        raise ValueError("business research requires exactly three queries")
    evidence = []
    for query in plan.queries:
        context = {
            "trace_id": trace_id, "node": view or "business_analysis", "view": view,
            "target_name": target_name,
            "attempt": attempt, "query": query, "rework_hints": list(plan.rework_hints),
        }
        with monitor("search", label="고객·현장 자료 검색", logger_name=__name__, **context) as metrics:
            results = search(query, max_results=MAX_RESULTS_PER_CALL)
            evidence.extend(results[:MAX_RESULTS_PER_CALL])
            metrics.update(result_count=len(results[:MAX_RESULTS_PER_CALL]),
                           source_ids=[item.source_id for item in results[:MAX_RESULTS_PER_CALL]],
                           source_urls=[item.url for item in results[:MAX_RESULTS_PER_CALL]])
    return dedupe_by_source_id([], evidence)
