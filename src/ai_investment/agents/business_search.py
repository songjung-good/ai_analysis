"""Bounded web research for the deployment and business Agent."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ..models import Evidence
from ..state import GraphState
from ..tools.web import web_search


MAX_SEARCH_CALLS = 3
MAX_RESULTS_PER_CALL = 5


@dataclass(frozen=True)
class SearchPlan:
    queries: tuple[str, ...]
    information_gaps: tuple[str, ...]


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


def build_search_plan(state: GraphState) -> SearchPlan:
    """Validate known input fields without mutating the shared State."""
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
    queries = (
        f"{subject}{customer} 고객 도입 유료 계약 시험 운영 customer deployment paid contract pilot",
        f"{subject} 가격 설치 유지보수 생산성 pricing maintenance productivity ROI case study",
        f"{subject} 안전 인증 현장 도입 safety certification deployment",
    )
    return SearchPlan(queries, tuple(dict.fromkeys(gaps)))


def collect_evidence(
    plan: SearchPlan, *, search: Callable[..., list[Evidence]] = web_search
) -> list[Evidence]:
    """Search once per topic; propagate failures and merge repeated source IDs."""
    if len(plan.queries) != MAX_SEARCH_CALLS:
        raise ValueError("business research requires exactly three queries")
    sources: dict[str, Evidence] = {}
    for query in plan.queries:
        results = search(query, max_results=MAX_RESULTS_PER_CALL)
        for item in results[:MAX_RESULTS_PER_CALL]:
            previous = sources.get(item.source_id)
            if previous is None:
                sources[item.source_id] = item
            elif item.excerpt not in previous.excerpt:
                sources[item.source_id] = previous.model_copy(
                    update={"excerpt": f"{previous.excerpt}\n\n{item.excerpt}"}
                )
    return list(sources.values())
