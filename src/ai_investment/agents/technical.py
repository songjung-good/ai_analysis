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
TECH_SOURCE_PREFIX = "tech:"  # ingest_tech_docs.py가 붙이는 논문·보고서 source_id 접두어


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
    company_specific: bool = Field(
        description="자료 중 이 기업의 제품·기술·성능을 직접 다룬 것이 하나라도 있는지"
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
    performance_evidence: list[Claim] = Field(description="이 기업 제품의 성능 근거만")
    benchmark_context: list[Claim] = Field(
        description="논문·보고서가 보여 주는 업계 기술 수준(다른 모델의 성능). 기업 성능으로 쓰지 않는다"
    )
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
- 한국어로 작성하되 VLA, foundation model 같은 핵심 영문 용어는 그대로 쓴다.
- 질의 하나에는 개념 하나만 담고, 20단어 이내의 자연어 문장으로 쓴다.
- AND, OR 같은 불리언 연산자나 키워드 나열은 쓰지 않는다 (의미 검색용 질의다)."""

GRADE_PROMPT = """아래 스타트업의 기술을 검증하려 한다.
검증 목적: 핵심 기술의 수준, 성능 근거, 기술적 한계 판단

[기업 정보]
{startup}

[검색 자료]
{context}

각 자료가 검증 목적에 실질적 근거가 되는지 판정하라.
- 관련 자료 ID만 relevant_ids에 넣는다.
- 핵심 기술, 성능 근거, 기술적 한계를 모두 판단할 수 있으면 sufficient=true.
- 이 기업의 제품·기술·성능을 직접 다룬 자료가 있으면 company_specific=true.
  다른 기업·모델의 논문이나 업계 일반 보고서만 있으면 false.
- 부족한 항목은 missing에 적는다."""

REWRITE_PROMPT = """다음 검색 질의로는 기술 검증 근거를 충분히 찾지 못했다.
의도는 유지하되 동의어와 영문 기술 용어(VLA, robot foundation model,
sim-to-real, generalization 등)를 섞어 질의 {n}개를 다시 작성하라.
질의 하나에는 개념 하나만 담고, 20단어 이내의 자연어 문장으로 쓴다.
AND, OR 같은 불리언 연산자는 쓰지 않는다.

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
- performance_evidence에는 이 기업 제품의 성능만 쓴다.
  논문 속 다른 모델(LingBot-VLA, τ0-VLA, π0 등)의 성능은 benchmark_context에 쓴다.
- limitations는 최소 1개 쓴다. 논문이 밝힌 기술 일반의 한계를 기업 기술에 비추어 서술해도 된다.
- summary 문장 안에 S1 같은 자료 ID를 쓰지 않는다.
- evidence_level은 이 기업 기술의 검증 수준이다.
  high: 제3자 자료(기사·고객 사례·인증)로 이 기업 제품의 성능이 확인됨
  medium: 기업 공개 자료로 기술 구성은 확인되나 제3자 성능 근거는 부족함
  low: 기업 구체 근거 없이 업계 일반 문헌으로만 판단함"""


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
    """registry가 이 Agent에 허용한 Tool(TOOLS)만 꺼낸다."""
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


@dataclass
class Grade:
    relevant: list[Evidence]
    sufficient: bool
    company_specific: bool
    missing: list[str]


def grade_relevance(startup: Mapping[str, Any], evidence: Sequence[Evidence]) -> Grade:
    if not evidence:
        return Grade([], False, False, ["검색 결과 없음"])
    labels = _label(evidence)
    grade = _structured(
        RelevanceGrade,
        GRADE_PROMPT.format(startup=_dump(startup), context=_context(labels)),
    )
    relevant = [labels[i] for i in dict.fromkeys(grade.relevant_ids) if i in labels]
    return Grade(relevant, grade.sufficient, grade.company_specific, list(grade.missing))


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
    """문헌 근거는 벡터DB에서, 기업 고유 근거가 없으면 웹에서 보완한다.

    - 관련 문헌이 부족하면 쿼리를 재작성해 벡터DB를 다시 검색한다 (최대 MAX_REWRITE회).
    - 재작성 후에도 부족하거나, 기업 제품을 직접 다룬 자료가 없으면 web_search로 폴백한다.
      (tech_docs는 업계 일반 논문·보고서라 기업 고유 근거는 대부분 웹에서 온다)
    """
    queries = generate_queries(startup, profile)
    result = CragResult(evidence=[])
    relevant: dict[str, Evidence] = {}
    docs_ok = False
    company_found = False

    for attempt in range(MAX_REWRITE + 1):
        found = retrieve(queries)
        result.queries.extend(queries)
        if attempt == 0 and not found:
            raise RuntimeError(
                "tech_docs collection이 비어 있습니다. "
                "먼저 python scripts/ingest_tech_docs.py 를 실행하세요."
            )
        grade = grade_relevance(startup, found)
        relevant.update((doc.source_id, doc) for doc in grade.relevant)
        company_found = company_found or grade.company_specific
        result.missing = grade.missing
        if grade.sufficient and len(relevant) >= MIN_RELEVANT_DOCS:
            docs_ok = True
            break
        if attempt < MAX_REWRITE:
            queries = rewrite_query(queries, grade.missing)
            result.rewrite_count += 1

    if not (docs_ok and company_found):
        result.used_web_fallback = True
        web = grade_relevance(startup, web_fallback(startup))
        relevant.update((doc.source_id, doc) for doc in web.relevant)
        result.missing = web.missing

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
                "benchmark_context": [],
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
        "benchmark_context": resolve(output.benchmark_context),
        "limitations": resolve(output.limitations),
        "evidence_level": output.evidence_level,
        "missing_items": list(output.missing_items),
    }
    # 업계 일반 문헌(tech_docs)만 인용했다면 기업 기술이 검증된 것이 아니므로 low로 제한한다.
    if not any(not ev.source_id.startswith(TECH_SOURCE_PREFIX) for ev in cited.values()):
        analysis["evidence_level"] = "low"
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
