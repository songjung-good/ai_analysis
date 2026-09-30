"""tech_docs collection 구축 스크립트 (설계서 B-2 기술·제품 검증 RAG)

사용법 (저장소 루트에서)
    PYTHONPATH=src python scripts/ingest_tech_docs.py            # 적재
    PYTHONPATH=src python scripts/ingest_tech_docs.py --dry-run  # 페이지 수만 확인

data/tech_docs/manifest.json 에 적힌 PDF만 적재한다.
- pages: "8-15" 또는 "1-12,20-22" 형식(1부터 시작, PDF 뷰어 기준 페이지). null이면 전체.
- 설계서 기준 전체 분량은 약 70p 이내로 맞춘다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_investment.tools.rag import TECH_COLLECTION, _store  # noqa: E402

DOCS_DIR = ROOT / "data" / "tech_docs"
MANIFEST = DOCS_DIR / "manifest.json"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
PAGE_BUDGET = 70
BATCH = 64


def parse_pages(spec: str | None, total: int) -> set[int]:
    if not spec:
        return set(range(1, total + 1))
    pages: set[int] = set()
    for part in spec.split(","):
        start, _, end = part.strip().partition("-")
        pages.update(range(int(start), int(end or start) + 1))
    return {p for p in pages if 1 <= p <= total}


def load(entry: dict):
    from langchain_community.document_loaders import PyMuPDFLoader

    path = DOCS_DIR / entry["file"]
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다. manifest의 file 이름과 맞추세요.")
    pages = PyMuPDFLoader(str(path)).load()
    keep = parse_pages(entry.get("pages"), len(pages))
    selected = [doc for doc in pages if int(doc.metadata.get("page", 0)) + 1 in keep]
    for doc in selected:
        # documents_to_evidence가 읽는 키만 정리한다. Chroma는 None 값을 받지 않는다.
        meta = {
            "source": entry["file"],
            "title": entry["title"],
            "page": int(doc.metadata.get("page", 0)),  # 0-based, Evidence에서 +1
        }
        for key in ("url", "published_at"):
            if entry.get(key):
                meta[key] = entry[key]
        doc.metadata = meta
    return selected, len(pages)


def split(documents):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    counters: dict[tuple[str, int], int] = {}
    for chunk in chunks:
        key = (chunk.metadata["source"], chunk.metadata["page"])
        counters[key] = counters.get(key, 0) + 1
        stem = Path(key[0]).stem
        chunk.metadata["source_id"] = f"tech:{stem}:p{key[1] + 1}:c{counters[key]}"
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="적재 없이 페이지 수만 출력")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")

    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    documents, used = [], 0
    for entry in entries:
        selected, total = load(entry)
        used += len(selected)
        documents.extend(selected)
        print(f"[load] {entry['file']}: 전체 {total}p 중 {len(selected)}p 사용")
    print(f"[total] {used}p (설계 기준 {PAGE_BUDGET}p 이내)")
    if used > PAGE_BUDGET:
        print("[warn] 분량 초과: manifest의 pages로 본문·한계 절 위주로 줄이세요.")
    if args.dry_run:
        return

    chunks = split(documents)
    store = _store(TECH_COLLECTION)
    store.reset_collection()  # 재실행 시 중복 적재 방지
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        store.add_documents(batch, ids=[c.metadata["source_id"] for c in batch])
    print(f"[done] collection={TECH_COLLECTION}, chunks={len(chunks)}")


if __name__ == "__main__":
    main()
