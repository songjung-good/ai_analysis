"""기술·제품 검증 Agent (설계서 B-1 3번, B-2, D-2 RAG 서브그래프)

CRAG 흐름 (설계서 D-2 RAG 서브그래프와 1:1 대응)

    질의 생성 -> 벡터DB 검색(search_tech_docs) -> 관련성 평가
        |- 충분                -> 분석 결과 생성
        |- 부족 & 재작성 가능  -> 쿼리 재작성 -> 벡터DB 검색
        '- 부족 & 재작성 초과  -> web_search 폴백 -> 분석 결과 생성

BASECODE 계약
- 쓰는 State 키: technical_analysis, references
- references에는 분석 결과에서 실제 인용한 Evidence만 넣는다.
- 재시도와 fallback은 이 Agent 안에서 끝내고, 예외는 숨기지 않는다.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field

from ..contracts import NodeResult
from ..models import Evidence
from ..state import GraphState
from ..tools import tools_for


AGENT = "technical_analysis"
TOOLS = tools_for(AGENT)

MAX_QUERIES = 3          # 질의 생성 개수
TECH_DOCS_K = 5          # 설계서 top_k=5 (fetch_k=20, lambda_mult=0.5는 search_tech_docs 내부)
MIN_RELEVANT_DOCS = 2    # 관련 자료가 이 개수 미만이면 "부족"
MAX_REWRITE = 1          # 쿼리 재작성 최대 횟수. 초과 시 web_search 폴백
WEB_MAX_RESULTS = 5
EXCERPT_CHARS = 900      # 프롬프트에 넣는 근거 길이 제한


# ---------------------------------------------------------------------------
# LLM structured output 스키마
# ---------------------------------------------------------------------------

Basis = Literal["company_claim", "third_party", "paper"]
Layer = Literal["perception", "decision", "action", "platform"]


class QueryPlan(BaseModel):
    queries: list[str] = Field(
        description="기술 문서·논문 벡터DB 검색 질의 3개. 기업명보다 기술 개념 중심."
    )


class RelevanceGrade(BaseModel):
    relevant_ids: list[str] = Field(description="검증 목적에 실질적 근거가 되는 자료 ID")
    sufficient: bool = Field(
        description="핵심 기술, 성능 근거, 기술적 한계를 판단하기에 자료가 충분한지"
    )
    missing: list[str] = Field(description="부족한 근거 항목. 충분하면 빈 목록")


class TechClaim(BaseModel):
    statement: str
    layer: Layer = Field(description="AI 기능 계층: 인식, 판단, 행동, 플랫폼")
    basis: Basis = Field(description="기업 주장, 제3자 자료, 논문 중 근거 유형")
    source_ids: list[str] = Field(description="인용한 자료 ID(S1, S2 ...)")


class Claim(BaseModel):
    statement: str
    basis: Basis
    source_ids: list[str]


class TechnicalAnalysisOutput(BaseModel):
    summary: str = Field(description="기술 검증 결과 3문장 이내 요약")
    core_technologies: list[TechClaim]
    differentiation: list[Claim]
    performance_evidence: list[Claim]
    limitations: list[Claim]
    evidence_level: Literal["high", "medium", "low"]
    missing_items: list[str]


# ---------------------------------------------------------------------------
# 프롬프트
# ---------------------------------------------------------------------------

QUERY_PROMPT = """당신은 Physical AI·Robotics 분야 기술 심사역이다.
아래 스타트업의 기술을 검증하기 위해 로봇 파운데이션 모델·VLA 논문과
기술 보고서가 들어 있는 벡터DB에서 찾을 검색 질의 {n}개를 만든다.

[기업 정보]
{startup}

[분야·고객 분류]
{profile}

규칙:
- 기업명이 아니라 기업 기술과 관련된 기술 개념으로 질의한다.
- AI의 인식·판단·행동 관점과 성능 근거, 기술적 한계를 골고루 포함한다.
- 한국어로 작성하되 VLA, foundation model 같은 핵심 영문 용어는 그대로 쓴다."""

GRADE_PROMPT = """아래 스타트업의 기술을 검증하려 한다.
검증 목적: 핵심 기술의 수준, 성능 근거, 기술적 한계 판단

[기업 정보]
{startup}

[검색 자료]
{context}

각 자료가 검증 목적에 실질적 근거가 되는지 판정하라.
- 관련 자료 ID만 relevant_ids에 넣는다.
- 핵심 기술, 성능 근거, 기술적 한계를 모두 판단할 수 있으면 sufficient=true.
- 부족한 항목은 missing에 적는다."""

REWRITE_PROMPT = """다음 검색 질의로는 기술 검증 근거를 충분히 찾지 못했다.
의도는 유지하되 동의어와 영문 기술 용어(VLA, robot foundation model,
sim-to-real, generalization 등)를 섞어 질의 {n}개를 다시 작성하라.

[기존 질의]
{queries}

[부족한 근거]
{missing}"""

ANALYZE_PROMPT = """당신은 Physical AI·Robotics 분야 기술 심사역이다.
아래 기업 정보와 근거 자료만 사용해 기업의 기술·제품을 검증하라.

[기업 정보]
{startup}

[분야·고객 분류]
{profile}

[근거 자료] 각 자료 앞의 ID(S1, S2 ...)로 인용한다.
{context}

작성 규칙:
- core_technologies는 AI 기능 계층(인식·판단·행동·플랫폼)을 구분해 쓴다.
- 기업의 주장(company_claim)과 제3자 자료(third_party), 논문(paper)을 구분한다.
- company_claim이 아닌 모든 주장에는 source_ids를 반드시 단다.
- 근거 자료에 없는 내용은 추측하지 말고 missing_items에 적는다.
- limitations는 최소 1개 쓴다. 논문이 밝힌 기술 일반의 한계를 기업 기술에 비추어 서술해도 된다.
- evidence_level: 제3자·논문 근거로 성능이 확인되면 high, 일부만 확인되면 medium,
  기업 주장뿐이면 low."""


# ---------------------------------------------------------------------------
# LLM·Tool 접근
# ---------------------------------------------------------------------------

T = TypeVar("T", bound=BaseModel)


@lru_cache(maxsize=1)
def _llm():
    from langchain_openai import ChatOpenAI

    model = os.getenv("OPENAI_MODEL")
    if not model:
        raise RuntimeError("OPENAI_MODEL 환경변수를 설정하세요")
    return ChatOpenAI(model=model, temperature=0)


def _structured(schema: type[T], prompt: str) -> T:
    return _llm().with_structured_output(schema).invoke(prompt)


def _tool(name: str):
    """registry가 이 Agent에 허용한 Tool만 꺼낸다."""
    tools = {tool.__name__: tool for tool in TOOLS}
    if name not in tools:
        raise PermissionError(f"{AGENT} is not allowed to use {name}")
    return tools[name]


# ---------------------------------------------------------------------------
# CRAG 단계
# ---------------------------------------------------------------------------

@dataclass
class CragResult:
    evidence: list[Evidence]
    queries: list[str] = field(default_factory=list)
    rewrite_count: int = 0
    used_web_fallback: bool = False
    missing: list[str] = field(default_factory=list)


def generate_queries(startup: Mapping[str, Any], profile: Mapping[str, Any]) -> list[str]:
    plan = _structured(
        QueryPlan,
        QUERY_PROMPT.format(n=MAX_QUERIES, startup=_dump(startup), profile=_dump(profile)),
    )
    return _clean_queries(plan.queries)


def retrieve(queries: Sequence[str]) -> list[Evidence]:
    search = _tool("search_tech_docs")
    return _dedupe(item for q in queries for item in search(q, k=TECH_DOCS_K))


def grade_relevance(
    startup: Mapping[str, Any], evidence: Sequence[Evidence]
) -> tuple[list[Evidence], bool, list[str]]:
    if not evidence:
        return [], False, ["검색 결과 없음"]
    labels = _label(evidence)
    grade = _structured(
        RelevanceGrade,
        GRADE_PROMPT.format(startup=_dump(startup), context=_context(labels)),
    )
    relevant = [labels[i] for i in dict.fromkeys(grade.relevant_ids) if i in labels]
    return relevant, grade.sufficient, list(grade.missing)


def rewrite_query(queries: Sequence[str], missing: Sequence[str]) -> list[str]:
    plan = _structured(
        QueryPlan,
        REWRITE_PROMPT.format(
            n=MAX_QUERIES,
            queries="\n".join(f"- {q}" for q in queries),
            missing="\n".join(f"- {m}" for m in missing) or "- (미상)",
        ),
    )
    return _clean_queries(plan.queries)


def web_fallback(startup: Mapping[str, Any]) -> list[Evidence]:
    name = str(startup.get("name", "")).strip()
    product = str(startup.get("product", "")).strip()
    search = _tool("web_search")
    queries = [
        f"{name} {product} 로봇 AI 핵심 기술 성능".replace("  ", " "),
        f"{name} robot AI technology benchmark limitation",
    ]
    return _dedupe(
        item for q in queries for item in search(q, max_results=WEB_MAX_RESULTS)
    )


def run_crag(startup: Mapping[str, Any], profile: Mapping[str, Any]) -> CragResult:
    queries = generate_queries(startup, profile)
    result = CragResult(evidence=[])
    relevant: dict[str, Evidence] = {}

    for attempt in range(MAX_REWRITE + 1):
        found = retrieve(queries)
        result.queries.extend(queries)
        if attempt == 0 and not found:
            raise RuntimeError(
                "tech_docs collection이 비어 있습니다. "
                "먼저 python scripts/ingest_tech_docs.py 를 실행하세요."
            )
        docs, sufficient, missing = grade_relevance(startup, found)
        relevant.update((doc.source_id, doc) for doc in docs)
        result.missing = missing
        if sufficient and len(relevant) >= MIN_RELEVANT_DOCS:
            result.evidence = list(relevant.values())
            return result
        if attempt < MAX_REWRITE:
            queries = rewrite_query(queries, missing)
            result.rewrite_count += 1

    # 부족 & 재작성 초과 -> 웹 검색 폴백
    result.used_web_fallback = True
    web_docs, _, missing = grade_relevance(startup, web_fallback(startup))
    relevant.update((doc.source_id, doc) for doc in web_docs)
    result.missing = missing
    result.evidence = list(relevant.values())
    return result


def analyze(
    startup: Mapping[str, Any],
    profile: Mapping[str, Any],
    evidence: Sequence[Evidence],
    missing: Sequence[str],
) -> tuple[dict[str, Any], list[Evidence]]:
    """분석 결과와 실제 인용된 Evidence를 반환한다."""
    if not evidence:
        # 근거 없이 LLM이 기술을 서술하면 환각이므로 분석하지 않는다.
        return (
            {
                "summary": "기술 문서와 웹 검색 모두에서 기술 검증 근거를 찾지 못했다.",
                "core_technologies": [],
                "differentiation": [],
                "performance_evidence": [],
                "limitations": [],
                "evidence_level": "insufficient",
                "missing_items": list(missing) or ["핵심 기술", "성능 근거", "기술적 한계"],
            },
            [],
        )

    labels = _label(evidence)
    output = _structured(
        TechnicalAnalysisOutput,
        ANALYZE_PROMPT.format(
            startup=_dump(startup), profile=_dump(profile), context=_context(labels)
        ),
    )

    cited: dict[str, Evidence] = {}

    def resolve(claims: Iterable[BaseModel]) -> list[dict[str, Any]]:
        resolved = []
        for claim in claims:
            data = claim.model_dump()
            valid = [label for label in data["source_ids"] if label in labels]
            if not valid and data["basis"] != "company_claim":
                continue  # 근거가 확인되지 않는 제3자·논문 주장은 버린다
            for label in valid:
                cited.setdefault(labels[label].source_id, labels[label])
            data["source_ids"] = [labels[label].source_id for label in valid]
            resolved.append(data)
        return resolved

    analysis = {
        "summary": output.summary,
        "core_technologies": resolve(output.core_technologies),
        "differentiation": resolve(output.differentiation),
        "performance_evidence": resolve(output.performance_evidence),
        "limitations": resolve(output.limitations),
        "evidence_level": output.evidence_level,
        "missing_items": list(output.missing_items),
    }
    return analysis, list(cited.values())


# ---------------------------------------------------------------------------
# Graph Node
# ---------------------------------------------------------------------------

def run(state: GraphState) -> NodeResult:
    """Validate core technology, evidence, and limitations with CRAG."""
    startup = state.get("selected_startup") or {}
    if not str(startup.get("name", "")).strip():
        raise ValueError("selected_startup.name is required")
    profile = state.get("profile") or {}

    crag = run_crag(startup, profile)
    analysis, cited = analyze(startup, profile, crag.evidence, crag.missing)
    analysis["startup_name"] = startup["name"]
    analysis["retrieval"] = {
        "queries": crag.queries,
        "rewrite_count": crag.rewrite_count,
        "used_web_fallback": crag.used_web_fallback,
        "relevant_count": len(crag.evidence),
        "cited_count": len(cited),
    }
    return {"technical_analysis": analysis, "references": cited}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _clean_queries(queries: Iterable[str]) -> list[str]:
    cleaned = [q.strip() for q in queries if q and q.strip()]
    if not cleaned:
        raise ValueError("LLM returned no search queries")
    return list(dict.fromkeys(cleaned))[:MAX_QUERIES]


def _dedupe(items: Iterable[Evidence]) -> list[Evidence]:
    return list({item.source_id: item for item in items}.values())


def _label(evidence: Sequence[Evidence]) -> dict[str, Evidence]:
    return {f"S{i}": item for i, item in enumerate(evidence, start=1)}


def _context(labels: Mapping[str, Evidence]) -> str:
    blocks = []
    for label, item in labels.items():
        where = f"p.{item.page}" if item.page else (item.url or "")
        blocks.append(f"[{label}] {item.title} {where}\n{item.excerpt[:EXCERPT_CHARS]}")
    return "\n\n".join(blocks)
