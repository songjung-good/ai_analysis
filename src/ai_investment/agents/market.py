from __future__ import annotations

import os
import re
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

MAX_FIGURES = 4
MAX_COMPETITORS = 6
# Startup map transcript line: "- 제조·산업 로봇 스타트업 (Robotics): A, B(비), ..."
MAP_LINE = re.compile(
    r"^- (?P<industry>\S+) (?P<tech>.+?) 스타트업 \(.+?\): (?P<names>.+)$", re.MULTILINE
)
# profile.subdomain (agents/profile.py CustomerProfile) -> startup map (industry, tech types).
# Subdomains the map does not classify (휴머노이드, 로봇 부품·하드웨어, 기타) fall back to the LLM.
MAP_SEGMENTS: dict[str, tuple[str | None, tuple[str, ...]]] = {
    "산업용·제조 로봇": ("제조·산업", ("로봇",)),
    "물류 로봇": ("물류·유통", ("로봇", "자율주행")),
    "자율주행·모빌리티": ("모빌리티·교통", ("자율주행",)),
    "서비스 로봇": ("서비스·생활", ("로봇",)),
    "로봇 AI 소프트웨어": (None, ("AI·SW 플랫폼",)),
}


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


class CompetitorDetail(BaseModel):
    name: str = Field(description="주어진 기업명 그대로")
    type: Literal["국내 스타트업", "해외 스타트업", "대기업·상장사"]
    product: str
    target_customer: str
    differentiator: str = Field(description="평가 대상 기업과 비교한 차이")


class CompetitorDetails(BaseModel):
    items: list[CompetitorDetail]


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


class LandscapeRow(BaseModel):
    industry: str
    tech_type: str
    companies: list[str]
    source_ids: list[str]


class CompetitorLandscape(BaseModel):
    """스타트업맵에서 코드로 뽑은 같은 분야 기업 전체. 같은 DB면 실행마다 같다."""

    segment: str | None = Field(description="'제조·산업 로봇'처럼 조회한 스타트업맵 분야, 대응 분야가 없으면 None")
    rows: list[LandscapeRow]
    company_count: int = Field(description="같은 분야 국내 스타트업 수(경쟁 강도 지표)")
    selected: list[str] = Field(description="competitors로 상세 분석한 기업, 선정 순서")
    evidence_sentences: dict[str, list[str]] = Field(
        description="선정 기업마다 코드가 근거에서 찾은 문장. LLM 요약과 무관하게 항상 같다"
    )
    other_mentioned: list[str] = Field(description="LLM이 근거에서 추가로 언급한 기업(검증됨, 상세 분석 제외)")


class TopicRetrieval(BaseModel):
    queries: list[str] = Field(description="실행한 검색 질의. 웹 검색은 'web:' 접두어")
    used_web: bool
    evidence_count: int


class Validation(BaseModel):
    dropped_uncited_items: int = Field(description="존재하지 않는 source_id만 인용해 제거한 항목 수")
    unverified_competitors: list[str] = Field(description="어떤 근거에도 이름이 없어 제거한 경쟁사")
    recited_competitors: list[str] = Field(description="인용을 이름이 실린 근거로 교정한 경쟁사")
    industry_mismatched_competitors: list[str] = Field(description="스타트업맵 산업 분류가 달라 제거한 경쟁사")
    unverified_figures: list[str] = Field(description="인용 근거에 값이 없어 제거한 수치")
    excluded_self_as_competitor: bool


class MarketAnalysisOutput(MarketAnalysis):
    """state["market_analysis"]의 계약. 투자 판단·보고서 Agent는 이 모델로 읽는다."""

    subdomain: str
    evidence_level: Literal["high", "medium", "low"] = Field(
        description="시장·수요·경쟁 3개 주제 중 근거가 충분한 주제 수: 3=high, 2=medium, 이하=low"
    )
    competitor_landscape: CompetitorLandscape
    validation: Validation
    retrieval: dict[Topic, TopicRetrieval]


def _known(value: object) -> str | None:
    """customer_profile writes "정보 부족" instead of guessing; treat it as absent."""
    text = str(value or "").strip()
    return None if not text or text == "정보 부족" else text


def build_questions(startup: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[Topic, str]:
    """질의 생성: profile의 세부 분야·고객·문제로 주제별 검색 질문을 만든다.

    industry·tech_type은 스타트업맵 분류(적용 산업 8개, 기술 유형 4개)를 따르면
    경쟁사 검색이 스타트업맵의 해당 칸과 바로 맞물린다.
    """
    subdomain = str(profile["subdomain"])
    customer = _known(profile.get("paying_customer")) or "주요 고객"
    problem = _known(profile.get("customer_problem")) or f"{subdomain} 자동화"
    resolved = resolve_segment(profile)
    segment = _segment_label(resolved) if resolved else subdomain
    return {
        "market": f"{subdomain} 시장 규모 설치 대수 성장률 전망",
        "demand": f"{customer} {problem} 수요 요인 인력 부족 비용 도입",
        "competition": f"{segment} 스타트업 경쟁사 {problem}",
    }


def resolve_segment(profile: Mapping[str, Any]) -> tuple[str | None, tuple[str, ...]] | None:
    """Explicit profile.industry/tech_type win over the subdomain mapping."""
    if profile.get("industry") or profile.get("tech_type"):
        tech = profile.get("tech_type")
        return profile.get("industry"), (str(tech),) if tech else ()
    return MAP_SEGMENTS.get(str(profile["subdomain"]))


def _segment_label(segment: tuple[str | None, tuple[str, ...]]) -> str:
    industry, techs = segment
    return " ".join(part for part in (industry, "·".join(techs)) if part)


def fetch_landscape(
    segment: tuple[str | None, tuple[str, ...]], search
) -> tuple[list[LandscapeRow], dict[str, Evidence], dict[str, Evidence]]:
    """Fixed-query lookups (no LLM): map rows of the segment, and narrative text about it.

    Returns (rows, map evidence, narrative evidence).
    """
    industry, techs = segment
    rows: dict[tuple[str, str], LandscapeRow] = {}
    map_evidence: dict[str, Evidence] = {}
    for tech in techs or ("",):
        query = " ".join(part for part in (industry, tech, "스타트업") if part)
        for item in search(query):
            for match in MAP_LINE.finditer(item.excerpt):
                if industry and match["industry"] != industry:
                    continue
                if techs and match["tech"] not in techs:
                    continue
                key = (match["industry"], match["tech"])
                if key not in rows:
                    names = [name.strip() for name in match["names"].split(",") if name.strip()]
                    rows[key] = LandscapeRow(
                        industry=key[0], tech_type=key[1], companies=names, source_ids=[item.source_id]
                    )
                    map_evidence[item.source_id] = item
    narrative_query = " ".join(part for part in (industry, *techs, "스타트업 투자 유치 사례") if part)
    narrative = {item.source_id: item for item in search(narrative_query) if not MAP_LINE.search(item.excerpt)}
    return list(rows.values()), map_evidence, narrative


def select_competitors(
    rows: Sequence[LandscapeRow], narrative: Mapping[str, Evidence], startup_name: str
) -> list[str]:
    """Companies also described in narrative text first, then map order; target excluded."""
    target = _name_keys(startup_name)
    candidates = [
        name for row in rows for name in row.companies if not (_name_keys(name) & target)
    ]
    mentions = {
        name: sum(_mentioned(name, item.excerpt) for item in narrative.values()) for name in candidates
    }
    ranked = sorted(range(len(candidates)), key=lambda i: (-mentions[candidates[i]], i))
    return [candidates[i] for i in ranked][:MAX_COMPETITORS]


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


def _name_keys(name: str) -> set[str]:
    """'Holiday Robotics(홀리데이로보틱스)' -> {'holidayrobotics', '홀리데이로보틱스'}."""
    parts = re.split(r"[()（）/,]", name)
    return {key for part in parts if len(key := re.sub(r"[\W_]+", "", part).lower()) >= 2}


def _mentioned(name: str, text: str) -> bool:
    compact = re.sub(r"[\W_]+", "", text).lower()
    return any(key in compact for key in _name_keys(name))


def _map_industries(name: str, evidence: Mapping[str, Evidence]) -> set[str]:
    """Industries the startup map assigns to a company, empty if it is not on the map."""
    return {
        match["industry"]
        for item in evidence.values()
        for match in MAP_LINE.finditer(item.excerpt)
        if _mentioned(name, match["names"])
    }


def _year_key(year: str) -> tuple[int, int]:
    """Prefer actual values over forecasts, then the latest year."""
    match = re.search(r"\d{4}", year)
    return ("전망" not in year, int(match.group()) if match else 0)


def _value_in_text(value: float, text: str) -> bool:
    """True if the number is written in the text as a whole number, not a digit fragment."""
    if value == int(value):
        spellings = {f"{int(value)}", f"{int(value):,}"}
    else:
        spellings = {f"{value:g}", f"{value:.1f}", f"{value:,.1f}", f"{value:.2f}"}
    return any(
        re.search(rf"(?<![\d.,]){re.escape(spelling)}(?![\d]|[.,]\d)", text)
        for spelling in spellings
    )


def _verified_figures(
    figures: Sequence[MarketFigure], evidence: Mapping[str, Evidence]
) -> tuple[list[MarketFigure], list[str]]:
    kept, dropped = [], []
    for figure in figures:
        cited_text = " ".join(evidence[sid].excerpt for sid in figure.source_ids if sid in evidence)
        if _value_in_text(figure.value, cited_text):
            kept.append(figure)
        else:
            dropped.append(f"{figure.metric} {figure.value:g}{figure.unit} ({figure.year})")
    return kept, dropped


def _latest_per_metric(figures: Sequence[MarketFigure]) -> list[MarketFigure]:
    best: dict[tuple[str, str, str], MarketFigure] = {}
    for figure in figures:
        key = (figure.metric, figure.region, figure.unit)
        if key not in best or _year_key(figure.year) > _year_key(best[key].year):
            best[key] = figure
    return list(best.values())[:MAX_FIGURES]


def curate(
    analysis: MarketAnalysis,
    evidence: Mapping[str, Evidence],
    startup_name: str,
    industry: str | None = None,
) -> tuple[MarketAnalysis, dict[str, Any]]:
    """Deterministic checks the LLM cannot be trusted with.

    - competitors must be named in the evidence they cite and must not be the target;
      a wrong citation is re-pointed to the retrieved evidence that names the company
    - a company the startup map files under another industry than profile.industry
      is not a competitor for the same customer
    - a figure's value must appear in the evidence it cites
    - one representative figure per metric, capped at MAX_FIGURES
    """
    target_keys = _name_keys(startup_name)
    competitors, unverified, recited, mismatched = [], [], [], []
    excluded_self = False
    for competitor in analysis.competitors:
        if _name_keys(competitor.name) & target_keys:
            excluded_self = True
            continue
        industries = _map_industries(competitor.name, evidence)
        if industry and industries and industry not in industries:
            mismatched.append(competitor.name)
            continue
        cited_text = " ".join(evidence[sid].excerpt for sid in competitor.source_ids if sid in evidence)
        if _mentioned(competitor.name, cited_text):
            competitors.append(competitor)
            continue
        naming = [sid for sid, item in evidence.items() if _mentioned(competitor.name, item.excerpt)]
        if naming:
            competitors.append(competitor.model_copy(update={"source_ids": naming}))
            recited.append(competitor.name)
        else:
            unverified.append(competitor.name)
    market_size, unverified_size = _verified_figures(analysis.market_size, evidence)
    growth, unverified_growth = _verified_figures(analysis.growth, evidence)
    curated = analysis.model_copy(
        update={
            "market_size": _latest_per_metric(market_size),
            "growth": _latest_per_metric(growth),
            "competitors": competitors,
        }
    )
    return curated, {
        "unverified_competitors": unverified,
        "recited_competitors": recited,
        "industry_mismatched_competitors": mismatched,
        "unverified_figures": unverified_size + unverified_growth,
        "excluded_self_as_competitor": excluded_self,
    }


SENTENCE_BREAK = re.compile(r"(?<=[.!?다])\s+|\n+")


def company_snippets(name: str, evidence: Mapping[str, Evidence]) -> list[tuple[str, str]]:
    """(source_id, sentence) pairs that name the company, excluding bare map lists."""
    snippets = []
    for sid, item in evidence.items():
        for sentence in SENTENCE_BREAK.split(item.excerpt):
            sentence = sentence.strip()
            if len(sentence) >= 10 and not MAP_LINE.match(sentence) and _mentioned(name, sentence):
                snippets.append((sid, sentence))
    return snippets


def _stub(name: str, source_ids: Sequence[str]) -> Competitor:
    return Competitor(
        name=name, type="국내 스타트업", product="근거 없음", target_customer="근거 없음",
        differentiator="근거 없음", source_ids=list(source_ids),
    )


def describe_competitors(
    llm,
    startup: Mapping[str, Any],
    selected: Sequence[str],
    rows: Sequence[LandscapeRow],
    evidence: Mapping[str, Evidence],
) -> list[Competitor]:
    """Selected companies in order. Code finds the sentences about each company; the LLM
    only rewrites those sentences. No sentences -> '근거 없음' without an LLM call."""
    row_sources = {name: row.source_ids for row in rows for name in row.companies}
    snippets = {name: company_snippets(name, evidence) for name in selected}
    described = [name for name in selected if snippets[name]]
    details: list[CompetitorDetail] = []
    if described:
        blocks = "\n\n".join(
            f"### {name}\n" + "\n".join(f"- {sentence}" for _, sentence in snippets[name])
            for name in described
        )
        details = llm.with_structured_output(CompetitorDetails).invoke(
            f"""평가 대상 기업 {startup["name"]}({startup.get("product") or "제품 정보 없음"})의 경쟁사를 정리합니다.
기업마다 아래 문장에 적힌 내용만으로 product, target_customer, differentiator를 채우세요.
문장에 없는 항목은 '근거 없음'으로 쓰고 지어내지 않습니다. name은 제목의 표기를 그대로 씁니다.
스타트업맵에 실린 기업은 '국내 스타트업'입니다.

{blocks}"""
        ).items
    competitors = []
    for name in selected:
        detail = next((d for d in details if _name_keys(d.name) & _name_keys(name)), None)
        cited = list(dict.fromkeys([*row_sources[name], *(sid for sid, _ in snippets[name])]))
        if detail is None:
            competitors.append(_stub(name, row_sources[name]))
        else:
            competitors.append(
                Competitor(**detail.model_dump(exclude={"name"}), name=name, source_ids=cited)
            )
    return competitors


def other_competitors(verified: Sequence[Competitor], selected: Sequence[str]) -> list[str]:
    """Verified competitors the main analysis named outside the code selection."""
    keys = [_name_keys(name) for name in selected]
    return [c.name for c in verified if not any(_name_keys(c.name) & k for k in keys)]


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
    selected: Sequence[str],
    landscape_evidence: Sequence[Evidence],
) -> str:
    sections = "\n\n".join(
        f"## {topic}: {result.question}\n{format_evidence(result.evidence) or '(근거 없음)'}"
        for topic, result in results.items()
    )
    if landscape_evidence:
        sections += f"\n\n## landscape: 스타트업맵 같은 분야 기업과 관련 본문\n{format_evidence(landscape_evidence)}"
    if selected:
        competitor_rule = (
            f"- 주요 경쟁사({', '.join(selected)})는 별도로 정리하므로 competitors에 넣지 않습니다. "
            "근거에 이 목록 밖의 경쟁사가 있을 때만 최대 3개를 competitors에 넣습니다."
        )
    else:
        competitor_rule = (
            "- 경쟁사는 최대 6개이며 같은 세부 분야의 국내 스타트업을 우선합니다. "
            "평가 대상 기업과 같은 고객 문제를 푸는 기업만 넣습니다."
        )
    return f"""당신은 Physical AI·Robotics 스타트업 투자 심사역입니다.
아래 근거만 사용해 평가 대상 기업의 시장성과 경쟁 구도를 분석하세요.

규칙:
- 모든 수치와 주장에는 근거의 [source_id]를 source_ids로 붙입니다. 근거에 없는 수치는 쓰지 않습니다.
- 수치는 근거에 적힌 값과 단위, 기준 연도를 그대로 옮기고 전망치는 year에 '(전망)'을 붙입니다.
{competitor_rule}
- 평가 대상 기업 자신은 경쟁사에 넣지 않습니다. 대기업 자회사·계열사·상장사는 '대기업·상장사'로 분류합니다.
- 세부 분야에 맞는 근거가 없으면 상위 시장 수치로 대체하되 metric에 상위 시장임을 밝히고 missing_info에 기록합니다.
- 판단할 근거가 없는 항목은 추정하지 말고 missing_info에 적습니다.
- market_size와 growth는 세부 분야와 가장 가까운 지표를 각각 최대 4개만 고릅니다. 같은 지표의 지역·연도별 나열은 대표값 하나로 줄입니다.
- 경쟁사 name은 근거에 적힌 표기를 그대로 씁니다.
- 스타트업맵에 실린 기업은 '국내 스타트업'입니다. 스타트업맵은 기업명만 담고 있으므로 다른 근거가 없으면 product·differentiator에 '근거 없음'이라고 쓰고 지어내지 않습니다.
- summary는 3문장, 300자 이내로 씁니다.

평가 대상 기업: {dict(startup)}
분야·고객 분류: {dict(profile)}

{sections}"""


# gpt-5-nano on PLAIF: default(medium) 168s, low ~50s with equal quality, minimal empty output.
DEFAULT_REASONING_EFFORT = "low"


@lru_cache(maxsize=1)
def _default_llm():
    from langchain_openai import ChatOpenAI

    model = os.getenv("OPENAI_MODEL")
    if not model:
        raise RuntimeError("OPENAI_MODEL is not set")
    # Non-reasoning models reject reasoning_effort; set OPENAI_REASONING_EFFORT= to omit it.
    effort = os.getenv("OPENAI_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
    return ChatOpenAI(model=model, **({"reasoning_effort": effort} if effort else {}))


def analyze(state: GraphState, *, llm, search=None, web=None) -> NodeResult:
    startup, profile = _validate_input(state)
    crag = build_crag(
        retrieve=search or (lambda query: search_market_docs(query, k=5)),
        grade=llm_grader(llm),
        rewrite=llm_rewriter(llm),
        web_search=web or (lambda query: web_search(query, max_results=5)),
    )
    retrieve = search or (lambda query: search_market_docs(query, k=5))
    segment = resolve_segment(profile)
    rows, map_evidence, narrative = fetch_landscape(segment, retrieve) if segment else ([], {}, {})
    selected = select_competitors(rows, narrative, str(startup["name"]))

    questions = build_questions(startup, profile)
    with ThreadPoolExecutor(max_workers=len(questions)) as pool:
        futures = {topic: pool.submit(run_crag, crag, q) for topic, q in questions.items()}
        results: dict[Topic, CragResult] = {t: f.result() for t, f in futures.items()}

    evidence: dict[str, Evidence] = {
        item.source_id: item for result in results.values() for item in result.evidence
    }
    landscape_evidence = {**map_evidence, **narrative}
    evidence.update(landscape_evidence)
    analysis = llm.with_structured_output(MarketAnalysis).invoke(
        _prompt(startup, profile, results, selected, list(landscape_evidence.values()))
    )
    analysis, dropped = enforce_citations(analysis, set(evidence))
    analysis, checks = curate(
        analysis, evidence, str(startup["name"]), segment[0] if segment else None
    )
    if selected:
        other_mentioned = other_competitors(analysis.competitors, selected)
        competitors = describe_competitors(llm, startup, selected, rows, evidence)
        sentences = {name: [text for _, text in company_snippets(name, evidence)] for name in selected}
    else:
        other_mentioned, sentences = [], {}
        competitors = analysis.competitors[:MAX_COMPETITORS]
    analysis = analysis.model_copy(update={"competitors": competitors})

    output = MarketAnalysisOutput(
        **analysis.model_dump(),
        subdomain=str(profile["subdomain"]),
        evidence_level=_evidence_level(results),
        competitor_landscape=CompetitorLandscape(
            segment=_segment_label(segment) if segment else None,
            rows=rows,
            company_count=sum(len(row.companies) for row in rows),
            selected=selected,
            evidence_sentences=sentences,
            other_mentioned=other_mentioned,
        ),
        validation=Validation(dropped_uncited_items=dropped, **checks),
        retrieval={
            topic: TopicRetrieval(
                queries=result.queries,
                used_web=result.used_web,
                evidence_count=len(result.evidence),
            )
            for topic, result in results.items()
        },
    )
    cited = _cited_ids(output) | {sid for row in rows for sid in row.source_ids}
    return {
        "market_analysis": output.model_dump(),
        "references": [item for sid, item in evidence.items() if sid in cited],
    }


def run(state: GraphState) -> NodeResult:
    """Assess market opportunity and competitors with CRAG."""
    return analyze(state, llm=_default_llm())
