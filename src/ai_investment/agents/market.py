from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..contracts import NodeResult
from ..crag import CragResult, build_crag, format_evidence, llm_grader, llm_rewriter, run_crag
from ..models import Evidence
from ..state import GraphState
from ..tools import tools_for


TOOLS = tools_for("market_analysis")
search_market_docs, web_search = TOOLS

Topic = Literal["market", "demand", "competition"]


class MarketFigure(BaseModel):
    metric: str = Field(description="지표명. 예: 산업용 로봇 연간 설치 대수, AI 로보틱스 시장 규모")
    value: float = Field(description="근거에 적힌 수치. 단위는 unit에 분리")
    unit: str = Field(description="예: 천 대, 억 달러, 억 원, %")
    year: str = Field(description="기준 연도 또는 기간. 전망치는 '2028(전망)'처럼 표기")
    region: str = Field(description="예: 세계, 한국, 중국")
    source_ids: list[str]


class Claim(BaseModel):
    claim: str
    source_ids: list[str]


class Competitor(BaseModel):
    name: str
    type: Literal["국내 스타트업", "해외 스타트업", "대기업·상장사"]
    product: str
    target_customer: str
    differentiator: str = Field(description="평가 대상 기업과 비교한 차이")
    source_ids: list[str]


class Differentiation(BaseModel):
    strengths: list[Claim]
    weaknesses: list[Claim]
    entry_barriers: list[Claim]


class MarketAnalysis(BaseModel):
    summary: str = Field(description="시장 기회와 경쟁 구도 3문장 이내 요약")
    market_size: list[MarketFigure]
    growth: list[MarketFigure] = Field(description="성장률, CAGR, 전망치")
    demand_drivers: list[Claim]
    competitors: list[Competitor]
    differentiation: Differentiation
    missing_info: list[str] = Field(description="근거를 찾지 못해 판단할 수 없는 항목")


def build_questions(startup: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[Topic, str]:
    """질의 생성: profile의 세부 분야·고객·문제로 주제별 검색 질문을 만든다.

    industry·tech_type은 스타트업맵 분류(적용 산업 8개, 기술 유형 4개)를 따르면
    경쟁사 검색이 스타트업맵의 해당 칸과 바로 맞물린다.
    """
    subdomain = str(profile["subdomain"])
    customer = profile.get("paying_customer") or "주요 고객"
    problem = profile.get("problem") or f"{subdomain} 자동화"
    segment = " ".join(
        str(value) for value in (profile.get("industry"), profile.get("tech_type")) if value
    ) or subdomain
    return {
        "market": f"{subdomain} 시장 규모 설치 대수 성장률 전망",
        "demand": f"{customer} {problem} 수요 요인 인력 부족 비용 도입",
        "competition": f"{segment} 국내 스타트업 경쟁사 {problem} 투자 유치",
    }


def _validate_input(state: GraphState) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    startup = state.get("selected_startup") or {}
    profile = state.get("profile") or {}
    if not startup.get("name"):
        raise ValueError("market_analysis requires selected_startup.name")
    if not profile.get("subdomain"):
        raise ValueError("market_analysis requires profile.subdomain")
    return startup, profile


def _evidence_level(results: Mapping[Topic, CragResult]) -> Literal["high", "medium", "low"]:
    sufficient = sum(result.sufficient for result in results.values())
    return "high" if sufficient == len(results) else "medium" if sufficient >= 2 else "low"


def _keep_cited(items: Sequence[Any], known: set[str]) -> tuple[list[Any], int]:
    """Drop hallucinated source_ids, then items left without any valid citation."""
    kept = []
    for item in items:
        valid = [sid for sid in item.source_ids if sid in known]
        if valid:
            kept.append(item.model_copy(update={"source_ids": valid}))
    return kept, len(items) - len(kept)


def enforce_citations(analysis: MarketAnalysis, known: set[str]) -> tuple[MarketAnalysis, int]:
    dropped = 0
    updates: dict[str, Any] = {}
    for field in ("market_size", "growth", "demand_drivers", "competitors"):
        updates[field], count = _keep_cited(getattr(analysis, field), known)
        dropped += count
    diff_updates = {}
    for field in ("strengths", "weaknesses", "entry_barriers"):
        diff_updates[field], count = _keep_cited(getattr(analysis.differentiation, field), known)
        dropped += count
    updates["differentiation"] = analysis.differentiation.model_copy(update=diff_updates)
    return analysis.model_copy(update=updates), dropped


def _cited_ids(analysis: MarketAnalysis) -> set[str]:
    items = [
        *analysis.market_size,
        *analysis.growth,
        *analysis.demand_drivers,
        *analysis.competitors,
        *analysis.differentiation.strengths,
        *analysis.differentiation.weaknesses,
        *analysis.differentiation.entry_barriers,
    ]
    return {sid for item in items for sid in item.source_ids}


def _prompt(
    startup: Mapping[str, Any],
    profile: Mapping[str, Any],
    results: Mapping[Topic, CragResult],
) -> str:
    sections = "\n\n".join(
        f"## {topic}: {result.question}\n{format_evidence(result.evidence) or '(근거 없음)'}"
        for topic, result in results.items()
    )
    return f"""당신은 Physical AI·Robotics 스타트업 투자 심사역입니다.
아래 근거만 사용해 평가 대상 기업의 시장성과 경쟁 구도를 분석하세요.

규칙:
- 모든 수치와 주장에는 근거의 [source_id]를 source_ids로 붙입니다. 근거에 없는 수치는 쓰지 않습니다.
- 수치는 근거에 적힌 값과 단위, 기준 연도를 그대로 옮기고 전망치는 year에 '(전망)'을 붙입니다.
- 경쟁사는 평가 대상 기업과 같은 고객 문제를 푸는 기업만 넣고, 평가 대상 기업 자신은 넣지 않습니다.
- 세부 분야에 맞는 근거가 없으면 상위 시장 수치로 대체하되 metric에 상위 시장임을 밝히고 missing_info에 기록합니다.
- 판단할 근거가 없는 항목은 추정하지 말고 missing_info에 적습니다.
- market_size와 growth는 세부 분야와 가장 가까운 지표를 각각 최대 4개만 고릅니다. 같은 지표의 지역·연도별 나열은 대표값 하나로 줄입니다.
- 경쟁사는 최대 6개이며 같은 세부 분야의 국내 스타트업을 우선합니다. 대기업 자회사·계열사·상장사는 '대기업·상장사'로 분류합니다.

평가 대상 기업: {dict(startup)}
분야·고객 분류: {dict(profile)}

{sections}"""


@lru_cache(maxsize=1)
def _default_llm():
    from langchain_openai import ChatOpenAI

    model = os.getenv("OPENAI_MODEL")
    if not model:
        raise RuntimeError("OPENAI_MODEL is not set")
    return ChatOpenAI(model=model)


def analyze(state: GraphState, *, llm, search=None, web=None) -> NodeResult:
    startup, profile = _validate_input(state)
    crag = build_crag(
        retrieve=search or (lambda query: search_market_docs(query, k=5)),
        grade=llm_grader(llm),
        rewrite=llm_rewriter(llm),
        web_search=web or (lambda query: web_search(query, max_results=5)),
    )
    questions = build_questions(startup, profile)
    with ThreadPoolExecutor(max_workers=len(questions)) as pool:
        futures = {topic: pool.submit(run_crag, crag, q) for topic, q in questions.items()}
        results: dict[Topic, CragResult] = {t: f.result() for t, f in futures.items()}

    evidence: dict[str, Evidence] = {
        item.source_id: item for result in results.values() for item in result.evidence
    }
    analysis = llm.with_structured_output(MarketAnalysis).invoke(
        _prompt(startup, profile, results)
    )
    analysis, dropped = enforce_citations(analysis, set(evidence))
    missing_info = list(analysis.missing_info)
    if dropped:
        missing_info.append(f"근거 인용이 없어 제거한 항목 {dropped}개")

    cited = _cited_ids(analysis)
    return {
        "market_analysis": {
            **analysis.model_dump(),
            "subdomain": profile["subdomain"],
            "missing_info": missing_info,
            "evidence_level": _evidence_level(results),
            "retrieval": {
                topic: {
                    "queries": result.queries,
                    "used_web": result.used_web,
                    "evidence_count": len(result.evidence),
                }
                for topic, result in results.items()
            },
        },
        "references": [item for sid, item in evidence.items() if sid in cited],
    }


def run(state: GraphState) -> NodeResult:
    """Assess market opportunity and competitors with CRAG."""
    return analyze(state, llm=_default_llm())
