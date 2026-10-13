from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging
import math
import os
import re
from threading import Lock
from time import monotonic, sleep
from typing import Any

from ..models import Evidence
from .common import source_id
from ..utils.logger import monitor


logger = logging.getLogger(__name__)
# ponytail: one process-wide lock; use a shared limiter if processes share a key.
_search_lock = Lock()
_next_search_at = 0.0
MAX_RATE_LIMIT_RETRIES = 2


class TavilyRateLimitError(ValueError):
    """Retries exhausted; ValueError prevents discovery from retrying again."""
    status_code = 429


def _rate_limited(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    if status is not None:
        return status == 429
    return bool(re.search(r"\b(?:Error\s+429|HTTP\s+429|429\s+Too Many Requests)\b", str(error), re.IGNORECASE))


def _seconds(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return value


def _results(payload: object) -> Sequence[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, Mapping):
        if "error" in payload:
            error = payload["error"]
            if isinstance(error, Exception):
                raise error
            raise RuntimeError(f"Tavily search returned an error: {error}")
        results = payload.get("results")
        if isinstance(results, list):
            return [item for item in results if isinstance(item, Mapping)]
    raise ValueError("Unexpected Tavily response: expected a results list")


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
    global _next_search_at
    if not query.strip():
        raise ValueError("query must not be empty")
    if not 1 <= max_results <= 5:
        raise ValueError("max_results must be between 1 and 5")
    interval = _seconds("TAVILY_MIN_INTERVAL_SECONDS", 1.0)
    backoff = _seconds("TAVILY_RETRY_BACKOFF_SECONDS", 15.0)

    from langchain_tavily import TavilySearch
    from langchain_core.tools import ToolException

    tool = TavilySearch(
        max_results=max_results, include_domains=domains or [], handle_tool_error=False
    )
    with monitor("search_queue", label="웹 검색 호출 순서 대기", query=query):
        _search_lock.acquire()
    try:
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            delay = max(0.0, _next_search_at - monotonic())
            if delay:
                with monitor("search_wait", label="웹 검색 호출 간격·재시도 대기", query=query,
                             attempt=attempt + 1, delay_seconds=round(delay, 2)):
                    sleep(delay)
            _next_search_at = monotonic() + interval
            try:
                with monitor("web_request", label="Tavily 검색 응답", query=query, attempt=attempt + 1) as metrics:
                    try:
                        payload = tool.invoke({"query": query})
                    except ToolException as exc:
                        if not str(exc).startswith("No search results found for '"):
                            raise
                        results = []
                    else:
                        results = _to_evidence(payload, max_results)
                    metrics.update(result_count=len(results), source_ids=[item.source_id for item in results],
                                   source_urls=[item.url for item in results],
                                   evidence_chars=sum(len(item.excerpt) for item in results))
                    return results
            except Exception as exc:
                if not _rate_limited(exc):
                    raise
                delay = backoff * (2 ** attempt)
                _next_search_at = max(_next_search_at, monotonic() + delay)
                if attempt == MAX_RATE_LIMIT_RETRIES:
                    raise TavilyRateLimitError(
                        "Tavily HTTP 429 persists after 3 attempts. "
                        "Wait before restarting and check key type/shared-key usage in Tavily Dashboard."
                    ) from exc
                logger.warning("Tavily HTTP 429: retry %s/%s after %.1fs", attempt + 1,
                               MAX_RATE_LIMIT_RETRIES, delay, extra={
                                   "node": "web_search", "event": "rate_limit_retry", "query": query,
                                   "retry_count": attempt + 1, "delay_seconds": delay,
                               })
    finally:
        _search_lock.release()
