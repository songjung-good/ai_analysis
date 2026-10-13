"""관점 정의와 하위 에이전트 공통 출력(ViewResult) 생성 도우미.

각 관점 담당자는 기존 분석 결과를 Claim 목록으로 바꾼 뒤 build_view_result()를 호출한다.
missing_items는 LLM이 쓰는 자유 목록이 아니라 "필수 확인 항목 중 인용 근거가 없는 항목"으로
코드가 계산한다. Supervisor의 충분성 판단이 이 값에 의존하므로 재현성이 보장된다.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import NotRequired, TypedDict

from ..state import VIEWS, View, ViewResult


class Claim(TypedDict):
    """state.Claim에 필수 항목 이름(item)과 근거 유형(basis)을 더한 형태."""

    text: str
    source_ids: list[str]
    item: NotRequired[str]        # 어떤 필수 확인 항목에 대한 주장인지
    basis: NotRequired[str]       # company_claim | third_party | paper ...


# supervisor가 라우팅할 그래프 노드 이름 (관점 이름 = 노드 이름)
VIEW_NODES: dict[View, str] = {view: view for view in VIEWS}

# 관점별 필수 확인 항목. 항목마다 인용(source_ids)이 붙은 주장이 1개 이상 있어야 covered.
REQUIRED_ITEMS: dict[View, tuple[str, ...]] = {
    "technical": ("핵심 기술", "성능 근거", "외부 검증", "기술 한계"),
    "market": ("시장 규모", "수요 근거", "경쟁 구도"),
    "stakeholder": ("고객·구매자", "공급·파트너", "이해 충돌·비용 부담"),
    "domain": ("적용 현장", "도입 사례·PoC", "운영 제약"),
}


def missing_required(
    view: View, claims: Iterable[Claim], extra_covered: Iterable[str] = ()
) -> list[str]:
    """인용된 주장이 하나도 없는 필수 항목을 순서대로 돌려준다."""
    covered = {c["item"] for c in claims if c.get("item") and c.get("source_ids")}
    covered.update(extra_covered)
    return [item for item in REQUIRED_ITEMS[view] if item not in covered]


def build_view_result(
    view: View,
    *,
    summary: str,
    claims: Sequence[Claim],
    extra_covered: Iterable[str] = (),
) -> ViewResult:
    source_ids = list(dict.fromkeys(sid for c in claims for sid in c.get("source_ids", [])))
    return {
        "view": view,
        "summary": summary,
        "claims": list(claims),
        "missing_items": missing_required(view, claims, extra_covered),
        "source_ids": source_ids,
        "status": "done",
    }


def failed_view_result(view: View, error: str, previous: ViewResult | None = None) -> ViewResult:
    """실행 실패. 이전 결과가 있으면 내용은 보존하되 status로 실패를 표시해 자동 통과를 막는다."""
    base: ViewResult = dict(previous) if previous else {  # type: ignore[assignment]
        "view": view,
        "summary": "",
        "claims": [],
        "missing_items": list(REQUIRED_ITEMS[view]),
        "source_ids": [],
    }
    # 오류 원문은 node_errors[view]에 남기고, 결과에는 요약만 함께 둔다.
    return {**base, "status": "failed", "error": error}  # type: ignore[typeddict-item]
