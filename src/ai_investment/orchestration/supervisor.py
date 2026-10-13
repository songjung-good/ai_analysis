"""Supervisor: 관점별 근거 충분성을 판단하고 다음에 실행할 하위 에이전트를 고른다.

라우팅은 순서 하드코딩이 아니라 매 호출마다 State(관점 결과, 충분성, 재작업 횟수,
평가 결과)를 보고 결정한다.

- 첫 호출: 아직 실행하지 않은 관점을 모두 반환 → 그래프가 병렬 실행
- 이후: 근거가 부족하거나 실패한 관점만 반환하고, 부족한 항목을 rework_hints로 전달
- 평가 노드가 research를 요청하면 해당 관점을 재조사 대상에 포함하고 eval_result를 비운다
  (요청을 소비했다는 표시. 같은 요청으로 반복 재조사하지 않는다)
- 반환할 관점이 없으면(전부 충분 또는 재작업 상한) resume_after_research로 진행
  (평소: investment_decision, 평가 노드 재조사 후: report_generation)

판단은 전부 코드 규칙이다(재현성). LLM은 하위 에이전트가 근거를 찾는 데만 쓴다.
결정과 사유는 State가 아니라 로그(trace_id 포함)로 남긴다.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

from ..models import Evidence
from ..state import VIEWS, GraphState, View, ViewResult
from .decision_log import log_decision
from .views import VIEW_NODES


MAX_REWORK = 2              # 관점별 재작업 상한 (최초 실행 제외)
MIN_DISTINCT_SOURCES = 2    # 서로 다른 출처 그룹 최소 개수
NEXT_AFTER_ANALYSIS = "investment_decision"
NEXT_AFTER_RESEARCH = "report_generation"


# ---------------------------------------------------------------------------
# 충분성 판단
# ---------------------------------------------------------------------------

def _evidence_index(references: list[Any] | None) -> dict[str, Evidence]:
    index: dict[str, Evidence] = {}
    for item in references or []:
        ev = item if isinstance(item, Evidence) else Evidence.model_validate(item)
        index[ev.source_id] = ev
    return index


def _source_group(source_id: str, index: Mapping[str, Evidence]) -> str:
    """같은 사이트·같은 문서는 하나의 출처로 센다.

    source_id는 청크 단위 해시라서 같은 논문의 다른 청크도 서로 다른 id를 가진다.
    """
    ev = index.get(source_id)
    if ev is None:
        return source_id
    if ev.url:
        host = urlparse(ev.url).netloc.lower().removeprefix("www.")
        if host:
            return host
    return ev.title.strip().lower()


def distinct_sources(result: ViewResult, references: list[Any] | None) -> int:
    index = _evidence_index(references)
    return len({_source_group(sid, index) for sid in result.get("source_ids", [])})


def sufficiency_gaps(result: ViewResult, references: list[Any] | None) -> list[str]:
    """충분하지 않은 이유 목록. 빈 목록이면 충분."""
    gaps = list(result.get("missing_items", []))
    n_sources = distinct_sources(result, references)
    if n_sources < MIN_DISTINCT_SOURCES:
        gaps.append(f"독립 출처 부족({n_sources}/{MIN_DISTINCT_SOURCES}): 다른 기관·매체의 근거")
    return gaps


def is_sufficient(result: ViewResult, references: list[Any] | None) -> bool:
    return result.get("status") == "done" and not sufficiency_gaps(result, references)


# ---------------------------------------------------------------------------
# Supervisor 노드
# ---------------------------------------------------------------------------

def _eval_research_targets(state: GraphState) -> list[View]:
    """평가 노드가 보낸 재조사 요청의 대상 관점. 처리하면 supervisor가 eval_result를 비운다."""
    verdict = state.get("eval_result")
    if not verdict or verdict.get("action") != "research":
        return []
    return [v for v in verdict.get("target_views", []) if v in VIEWS]


def _eval_hints(state: GraphState, view: View) -> list[str]:
    verdict = state.get("eval_result") or {}
    specific = (verdict.get("research_hints") or {}).get(view)
    if specific:
        return list(specific)
    reasons = [
        f"품질 평가 {name}: {item.get('reason', '')}"
        for name, item in (verdict.get("items") or {}).items()
        if not item.get("passed", True)
    ]
    return reasons or ["품질 평가 노드 재조사 요청"]


def supervisor(state: GraphState) -> dict[str, Any]:
    analyses: Mapping[str, ViewResult] = state.get("analyses") or {}
    references = state.get("references")
    counts = dict(state.get("rework_count") or {})
    step = state.get("step_count", 0)
    max_steps = state.get("max_steps", 20)

    todo: list[View] = []
    hints: dict[str, list[str]] = {}
    reasons: dict[str, str] = {}
    update: dict[str, Any] = {"step_count": step + 1}

    if step >= max_steps:
        reasons["_guard"] = f"supervisor 호출 상한 도달({step}/{max_steps})"
    else:
        for view in VIEWS:
            result = analyses.get(view)
            used = counts.get(view, 0)
            if result is None:
                todo.append(view)
                reasons[view] = "미실행"
                continue
            if result.get("status") == "failed":
                if used < MAX_REWORK:
                    todo.append(view)
                    counts[view] = used + 1
                    reasons[view] = f"실행 실패 재시도: {result.get('error', '')}"
                else:
                    reasons[view] = "실행 실패, 재시도 상한 도달 → 근거 부족으로 진행"
                continue
            gaps = sufficiency_gaps(result, references)
            if not gaps:
                reasons[view] = "충분"
            elif used < MAX_REWORK:
                todo.append(view)
                counts[view] = used + 1
                hints[view] = gaps
                reasons[view] = f"근거 부족 → 재작업 {used + 1}/{MAX_REWORK}: {gaps}"
            else:
                reasons[view] = f"근거 부족, 재작업 상한 도달 → 한계로 기록: {gaps}"

        targets = _eval_research_targets(state)
        if targets:
            update["eval_result"] = None  # 요청 소비 → 다음 supervisor 호출에서 반복하지 않음
            for view in targets:
                used = counts.get(view, 0)
                if view in todo:
                    hints.setdefault(view, []).extend(_eval_hints(state, view))
                elif used < MAX_REWORK:
                    todo.append(view)
                    counts[view] = used + 1
                    hints[view] = _eval_hints(state, view)
                    reasons[view] = f"품질 평가 재조사 요청 → 재작업 {used + 1}/{MAX_REWORK}"
                else:
                    reasons[view] = "품질 평가 재조사 요청, 재작업 상한 도달"

    update.update(
        next_views=todo,
        rework_count=counts,
        # 이번에 보낸 관점의 지시만 남긴다. 지난 라운드 지시가 다음 실행에 섞이지 않게 한다.
        rework_hints={view: hints.get(view, []) for view in VIEWS},
    )
    _log_decision(state, step, todo, reasons)
    return update


def _resume_target(state: GraphState) -> str:
    return state.get("resume_after_research") or NEXT_AFTER_ANALYSIS


def route_from_supervisor(state: GraphState) -> list[str] | str:
    """add_conditional_edges용. 관점 노드 목록(병렬) 또는 다음 단계 노드."""
    views = state.get("next_views") or []
    return [VIEW_NODES[v] for v in views] or _resume_target(state)


# 그래프 조립 시 path_map으로 사용
SUPERVISOR_DESTINATIONS = [*VIEW_NODES.values(), NEXT_AFTER_ANALYSIS, NEXT_AFTER_RESEARCH]


def _log_decision(state: GraphState, step: int, todo: list[View], reasons: dict[str, str]) -> None:
    log_decision(
        state.get("trace_id"),
        "supervisor",
        todo or _resume_target(state),
        {
            "step": step,
            "startup": (state.get("selected_startup") or {}).get("name"),
            **reasons,
        },
    )
