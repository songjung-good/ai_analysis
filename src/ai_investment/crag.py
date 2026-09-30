"""Corrective RAG subgraph shared by the technical and market agents.

query -> vector search -> grade_relevance
  -> sufficient: done
  -> insufficient and rewrites left: rewrite_query -> vector search
  -> insufficient and no rewrites left: web_search fallback -> done
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

from .models import Evidence


Retriever = Callable[[str], list[Evidence]]
Grader = Callable[[str, Sequence[Evidence]], set[str]]
Rewriter = Callable[[str, str, Sequence[Evidence]], str]

MIN_RELEVANT = 2
MAX_REWRITES = 1


class CragState(TypedDict, total=False):
    question: str
    query: str
    rewrites: int
    candidates: list[Evidence]
    relevant: Annotated[list[Evidence], operator.add]
    queries: Annotated[list[str], operator.add]
    used_web: bool


@dataclass(frozen=True)
class CragResult:
    question: str
    evidence: list[Evidence]
    queries: list[str]
    used_web: bool

    @property
    def sufficient(self) -> bool:
        return len(self.evidence) >= MIN_RELEVANT


class RelevanceGrade(BaseModel):
    relevant_ids: list[str] = Field(
        description="질문에 답하는 데 직접 쓸 수 있는 근거의 source_id 목록"
    )


class RewrittenQuery(BaseModel):
    query: str = Field(description="검색 엔진에 넣을 15단어 이내 키워드 질의")


def _dedupe(evidence: Sequence[Evidence]) -> list[Evidence]:
    seen: dict[str, Evidence] = {}
    for item in evidence:
        seen.setdefault(item.source_id, item)
    return list(seen.values())


def format_evidence(evidence: Sequence[Evidence], max_chars: int = 700) -> str:
    blocks = []
    for item in evidence:
        location = f"p.{item.page}" if item.page else (item.url or "")
        blocks.append(
            f"[{item.source_id}] {item.title} {location} ({item.published_at or '발행일 미상'})\n"
            f"{item.excerpt[:max_chars]}"
        )
    return "\n\n".join(blocks)


def llm_grader(llm) -> Grader:
    """Build grade_relevance from a chat model supporting structured output."""
    structured = llm.with_structured_output(RelevanceGrade)

    def grade_relevance(question: str, evidence: Sequence[Evidence]) -> set[str]:
        if not evidence:
            return set()
        result = structured.invoke(
            "다음 근거 중 질문에 답하는 데 직접 사용할 수 있는 것만 고르세요. "
            "주제만 비슷하고 질문의 대상·지표에 답하지 못하면 제외합니다.\n\n"
            f"질문: {question}\n\n근거:\n{format_evidence(evidence)}"
        )
        return set(result.relevant_ids) & {item.source_id for item in evidence}

    return grade_relevance


def llm_rewriter(llm) -> Rewriter:
    """Build rewrite_query from a chat model supporting structured output."""
    structured = llm.with_structured_output(RewrittenQuery)

    def rewrite_query(question: str, query: str, evidence: Sequence[Evidence]) -> str:
        result = structured.invoke(
            "이전 검색 질의로 질문에 맞는 근거를 충분히 찾지 못했습니다. "
            "동의어, 영문 표기, 상위·하위 개념을 활용해 다른 각도의 검색 질의를 만드세요. "
            "요청문이 아닌 핵심 키워드 나열로, 15단어 이내로 씁니다.\n\n"
            f"질문: {question}\n이전 질의: {query}\n"
            f"이전 결과 제목: {', '.join(item.title for item in evidence) or '없음'}"
        )
        if not result.query.strip():
            raise ValueError("rewrite_query returned an empty query")
        return result.query.strip()

    return rewrite_query


def build_crag(
    *,
    retrieve: Retriever,
    grade: Grader,
    rewrite: Rewriter,
    web_search: Retriever,
    max_rewrites: int = MAX_REWRITES,
):
    """Compile the CRAG subgraph from injected search and LLM functions."""
    from langgraph.graph import END, START, StateGraph

    def search_vector(state: CragState) -> dict:
        return {"candidates": retrieve(state["query"]), "queries": [state["query"]]}

    def grade_candidates(state: CragState) -> dict:
        relevant_ids = grade(state["question"], state["candidates"])
        return {
            "relevant": [e for e in state["candidates"] if e.source_id in relevant_ids]
        }

    def route(state: CragState) -> Literal["done", "rewrite", "web"]:
        if len(_dedupe(state.get("relevant", []))) >= MIN_RELEVANT:
            return "done"
        return "rewrite" if state.get("rewrites", 0) < max_rewrites else "web"

    def rewrite_query(state: CragState) -> dict:
        query = rewrite(state["question"], state["query"], state["candidates"])
        return {"query": query, "rewrites": state.get("rewrites", 0) + 1}

    def search_web(state: CragState) -> dict:
        results = web_search(state["query"])
        relevant_ids = grade(state["question"], results)
        return {
            "relevant": [e for e in results if e.source_id in relevant_ids],
            "queries": [f"web:{state['query']}"],
            "used_web": True,
        }

    builder = StateGraph(CragState)
    builder.add_node("search_vector", search_vector)
    builder.add_node("grade_relevance", grade_candidates)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("web_search", search_web)
    builder.add_edge(START, "search_vector")
    builder.add_edge("search_vector", "grade_relevance")
    builder.add_conditional_edges(
        "grade_relevance",
        route,
        {"done": END, "rewrite": "rewrite_query", "web": "web_search"},
    )
    builder.add_edge("rewrite_query", "search_vector")
    builder.add_edge("web_search", END)
    return builder.compile()


def run_crag(graph, question: str, query: str | None = None) -> CragResult:
    final = graph.invoke(
        {"question": question, "query": query or question, "rewrites": 0, "used_web": False}
    )
    return CragResult(
        question=question,
        evidence=_dedupe(final.get("relevant", [])),
        queries=final.get("queries", []),
        used_web=final.get("used_web", False),
    )
