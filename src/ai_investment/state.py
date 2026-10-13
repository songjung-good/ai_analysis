from __future__ import annotations

import operator
from typing import Annotated, Literal, NotRequired, TypeVar, TypedDict
from uuid import uuid4

from .models import Evidence


Decision = Literal["invest", "conditional", "hold"]
View = Literal["technical", "market", "stakeholder", "domain"]
VIEWS: tuple[View, ...] = ("technical", "market", "stakeholder", "domain")
K = TypeVar("K")
T = TypeVar("T")


def merge_dict(previous: dict[K, T], update: dict[K, T]) -> dict[K, T]:
    """Keep unrelated keys and replace a key with its newest result."""
    return {**previous, **update}


def dedupe_by_source_id(previous: list[Evidence], update: list[Evidence]) -> list[Evidence]:
    """Keep one source record per ID without losing previously cited excerpts."""
    sources: dict[str, Evidence] = {}
    for item in [*previous, *update]:
        existing = sources.get(item.source_id)
        if existing is None:
            sources[item.source_id] = item
        elif item.excerpt not in existing.excerpt:
            sources[item.source_id] = existing.model_copy(
                update={"excerpt": f"{existing.excerpt}\n\n{item.excerpt}"}
            )
    return list(sources.values())


class Claim(TypedDict):
    text: str
    source_ids: list[str]


class ViewResult(TypedDict):
    view: View
    summary: str
    claims: list[Claim]
    missing_items: list[str]
    source_ids: list[str]
    status: Literal["done", "failed"]


class EvalItem(TypedDict):
    passed: bool
    reason: str


class EvalVerdict(TypedDict):
    passed: bool
    items: dict[str, EvalItem]
    action: Literal["pass", "rewrite", "research"]
    target_views: list[View]
    feedback: list[str]
    research_hints: NotRequired[dict[View, list[str]]]


class Startup(TypedDict, total=False):
    name: str
    product: str
    funding_stage: str
    team: list[str]
    customer_application: str
    source_ids: list[str]


class Evaluation(TypedDict):
    startup_name: str
    score: float | None
    decision: Decision
    reason: str
    score_details: NotRequired[dict[str, dict[str, object]]]
    blocking_risks: NotRequired[list[dict[str, object]]]
    missing_information: NotRequired[list[str]]
    source_ids: NotRequired[list[str]]


class GraphState(TypedDict, total=False):
    # 작업 페이로드: 기존 분석 필드는 투자 판단과 기존 그래프 호환용으로 유지한다.
    domain: str
    criteria: dict[str, object]
    candidates: list[Startup]
    selected_startup: Startup
    profile: dict[str, object]
    technical_analysis: dict[str, object]
    business_analysis: dict[str, object]
    market_analysis: dict[str, object]
    analyses: Annotated[dict[View, ViewResult], merge_dict]
    scores: dict[str, float]
    investment_score: float | None
    decision: Decision | None
    decision_reason: str
    evaluations: Annotated[list[Evaluation], operator.add]
    references: Annotated[list[Evidence], dedupe_by_source_id]
    report_draft: dict[str, object] | None
    report: str
    eval_result: EvalVerdict | None

    # 제어 메타: 현재 작업과 종료 판단에 필요한 값만 저장한다.
    # 결정 로그 본문은 외부에 두고 trace_id로 실행과 연결한다.
    trace_id: str
    current_idx: int
    max_iterations: int
    next_views: list[View]
    resume_after_research: Literal["investment_decision", "report_generation"]
    rework_count: dict[View, int]
    rework_hints: dict[View, list[str]]
    step_count: int
    max_steps: int
    report_attempts: int  # 평가 미달 횟수. 최초 초안 이후 보완 2회를 허용하면 3번째 미달에 종료.
    node_status: Annotated[dict[str, Literal["pending", "done", "failed"]], merge_dict]
    # 같은 관점 ID로 상태와 오류를 조회한다. 성공하면 해당 관점 오류만 None으로 갱신한다.
    node_errors: Annotated[dict[str, str | None], merge_dict]


def candidate_reset_update() -> dict[str, object]:
    """Return the new-candidate reset contract for the integration node.

    Apply only after prior parallel work has joined. References, evaluations,
    trace_id and the global supervisor step budget remain run-wide.
    """
    from langgraph.types import Overwrite

    return {
        "analyses": Overwrite({}),
        "node_status": Overwrite({}),
        "node_errors": Overwrite({}),
        "profile": {},
        "technical_analysis": {},
        "business_analysis": {},
        "market_analysis": {},
        "scores": {},
        "investment_score": None,
        "decision": None,
        "decision_reason": "",
        "next_views": [],
        "resume_after_research": "investment_decision",
        "rework_count": {view: 0 for view in VIEWS},
        "rework_hints": {view: [] for view in VIEWS},
        "eval_result": None,
        "report_draft": None,
        "report_attempts": 0,
        "report": "",
    }


def create_initial_state(
    *, domain: str, criteria: dict[str, object], max_iterations: int,
    max_steps: int = 20, trace_id: str | None = None,
) -> GraphState:
    """Create validated input, empty results, and per-run control metadata."""
    if not domain.strip():
        raise ValueError("domain must not be empty")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if trace_id is not None and not trace_id.strip():
        raise ValueError("trace_id must not be empty")
    return {
        "domain": domain,
        "criteria": criteria,
        "max_iterations": max_iterations,
        "evaluations": [],
        "references": [],
        "analyses": {},
        "report_draft": None,
        "eval_result": None,
        "trace_id": trace_id if trace_id is not None else uuid4().hex,
        "next_views": [],
        "resume_after_research": "investment_decision",
        "rework_count": {view: 0 for view in VIEWS},
        "rework_hints": {view: [] for view in VIEWS},
        "step_count": 0,
        "max_steps": max_steps,
        "report_attempts": 0,
        "node_status": {},
        "node_errors": {},
    }
