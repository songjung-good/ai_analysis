from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..models import Evidence
from .common import source_id


def _results(payload: object) -> Sequence[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, Mapping):
        if "error" in payload:
            error = payload["error"]
            if isinstance(error, Exception):
                raise error
            raise RuntimeError("Tavily search returned an error")
        results = payload.get("results", [])
        if isinstance(results, list):
            return [item for item in results if isinstance(item, Mapping)]
    return []


def _to_evidence(payload: object, max_results: int) -> list[Evidence]:
    evidence = []
    for item in _results(payload)[:max_results]:
        url = str(item.get("url") or "") or None
        title = str(item.get("title") or url or "Untitled")
        excerpt = str(item.get("content") or item.get("raw_content") or "").strip()
        if not excerpt:
            continue
        evidence.append(
            Evidence(
                source_id=source_id(url, title),
                title=title,
                url=url,
                excerpt=excerpt,
                published_at=item.get("published_date") or item.get("published_at"),
            )
        )
    return evidence


def web_search(
    query: str, domains: list[str] | None = None, max_results: int = 5
) -> list[Evidence]:
    if not query.strip():
        raise ValueError("query must not be empty")
    if not 1 <= max_results <= 5:
        raise ValueError("max_results must be between 1 and 5")

    from langchain_tavily import TavilySearch

    tool = TavilySearch(max_results=max_results, include_domains=domains or [])
    return _to_evidence(tool.invoke({"query": query}), max_results)
