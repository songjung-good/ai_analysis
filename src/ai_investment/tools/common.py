from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

from ..models import Evidence


def source_id(*parts: object) -> str:
    value = "|".join(str(part or "") for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def documents_to_evidence(documents: Iterable[Any]) -> list[Evidence]:
    evidence = []
    for document in documents:
        metadata: Mapping[str, Any] = document.metadata or {}
        title = str(metadata.get("title") or metadata.get("source") or "Untitled")
        url = metadata.get("url")
        page = metadata.get("page")
        if page is not None:
            page = int(page) + 1 if metadata.get("page_zero_based", True) else int(page)
        evidence.append(
            Evidence(
                source_id=str(
                    metadata.get("source_id")
                    or source_id(url or metadata.get("source"), page, document.page_content)
                ),
                title=title,
                url=str(url) if url else None,
                page=page,
                excerpt=document.page_content.strip(),
                published_at=metadata.get("published_at"),
            )
        )
    return evidence
