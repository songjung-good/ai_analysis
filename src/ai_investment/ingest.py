"""Build the local Chroma collections used by the RAG search tools.

Usage:
    PYTHONPATH=src python -m ai_investment.ingest market
    PYTHONPATH=src python -m ai_investment.ingest market --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tools.common import source_id
from .tools.rag import MARKET_COLLECTION, TECH_COLLECTION


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MANIFEST_KEYS = ("title", "publisher", "published_at", "url")
TRANSCRIPT_PAGE = re.compile(r"^## p\.(\d+)( replace)?\s*$", re.MULTILINE)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
MEANINGFUL_CHAR = re.compile(r"[0-9A-Za-z가-힣]")
# Transcript sections (## / ### / ####) are split first so one chunk does not mix
# sections: startup map p.7 buried "맺으며" under "(4) AI·SW 플랫폼" (rank 3 -> 1).
SEPARATORS = ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""]
# PDF control characters used as spaces (SPRi headers) and private-use bullet glyphs
NOISE_CHAR = re.compile(r"[\x00-\x08\x0b-\x1f\x7f\ue000-\uf8ff]")


@dataclass(frozen=True)
class CollectionSpec:
    name: str
    root: Path
    chunk_size: int
    chunk_overlap: int

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def transcript_dir(self) -> Path:
        return self.root / "transcripts"

    @property
    def manifest_path(self) -> Path:
        return self.root / "sources.json"


COLLECTIONS = {
    "market": CollectionSpec(MARKET_COLLECTION, DATA_DIR / "market", 1000, 150),
    "tech": CollectionSpec(TECH_COLLECTION, DATA_DIR / "tech", 800, 100),
}


@dataclass(frozen=True)
class TranscriptPage:
    text: str
    replace: bool = False


def parse_transcript(text: str) -> dict[int, TranscriptPage]:
    """Map 0-based page index to manual transcript (`## p.N` is 1-based).

    `## p.N replace` discards the PDF text layer of that page, for pages whose
    extracted text is garbled rather than missing.
    """
    text = HTML_COMMENT.sub("", text)
    matches = list(TRANSCRIPT_PAGE.finditer(text))
    pages: dict[int, TranscriptPage] = {}
    for current, following in zip(matches, [*matches[1:], None]):
        page_number = int(current.group(1))
        if page_number < 1:
            raise ValueError(f"transcript page must start at 1: p.{page_number}")
        if page_number - 1 in pages:
            raise ValueError(f"duplicate transcript page: p.{page_number}")
        end = following.start() if following else len(text)
        body = text[current.end() : end].strip()
        if body:
            pages[page_number - 1] = TranscriptPage(body, bool(current.group(2)))
    return pages


def has_text_layer(text: str) -> bool:
    """Reject empty or garbled extraction such as '\\x01\\x01' or '!!!Korean!Map'."""
    stripped = "".join(text.split())
    if not stripped:
        return False
    meaningful = len(MEANINGFUL_CHAR.findall(stripped))
    return meaningful >= 30 and meaningful / len(stripped) >= 0.6


def merge_page_text(
    pdf_text: str, transcript: TranscriptPage | None
) -> tuple[str, str]:
    """Return page text and how it was extracted."""
    pdf_text = NOISE_CHAR.sub(" ", pdf_text)
    if transcript is None:
        return pdf_text.strip(), "pdf"
    if not transcript.replace and has_text_layer(pdf_text):
        return f"{pdf_text.strip()}\n\n{transcript.text}", "pdf+transcript"
    return transcript.text, "transcript"


def load_manifest(spec: CollectionSpec) -> dict[str, dict[str, Any]]:
    if not spec.manifest_path.exists():
        raise FileNotFoundError(f"missing source manifest: {spec.manifest_path}")
    manifest = json.loads(spec.manifest_path.read_text(encoding="utf-8"))
    pdfs = {path.name for path in spec.raw_dir.glob("*.pdf")}
    absent = sorted(manifest.keys() - pdfs)
    if absent:
        raise FileNotFoundError(
            f"Missing source PDFs in {spec.raw_dir}: {', '.join(absent)}. "
            "Create this directory and place the original PDFs there using these exact filenames. "
            "transcripts/ supplements PDF text and cannot replace the original PDFs. "
            "See README.md for RAG setup."
        )
    if not pdfs:
        raise FileNotFoundError(f"no PDF files in {spec.raw_dir}; see README.md for RAG setup")
    missing = sorted(pdfs - manifest.keys())
    if missing:
        raise ValueError(f"PDFs missing from {spec.manifest_path}: {', '.join(missing)}")
    for name, entry in manifest.items():
        if not entry.get("title"):
            raise ValueError(f"manifest entry needs a title: {name}")
    return manifest


def load_documents(spec: CollectionSpec) -> list[Any]:
    """Load one Document per page with manual transcripts merged in."""
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_core.documents import Document

    documents = []
    for file_name, entry in sorted(load_manifest(spec).items()):
        pdf_path = spec.raw_dir / file_name
        transcript_path = spec.transcript_dir / f"{pdf_path.stem}.md"
        transcripts = (
            parse_transcript(transcript_path.read_text(encoding="utf-8"))
            if transcript_path.exists()
            else {}
        )
        pages = PyMuPDFLoader(str(pdf_path)).load()
        out_of_range = sorted(index + 1 for index in transcripts if index >= len(pages))
        if out_of_range:
            raise ValueError(f"{transcript_path} has pages beyond the PDF: {out_of_range}")

        base_metadata = {key: entry.get(key) for key in MANIFEST_KEYS}
        for page in pages:
            index = int(page.metadata["page"])
            text, extraction = merge_page_text(page.page_content, transcripts.get(index))
            if not has_text_layer(text):
                continue
            metadata = {
                **base_metadata,
                "source": pdf_path.as_posix(),
                "page": index,
                "extraction": extraction,
            }
            documents.append(
                Document(
                    page_content=text,
                    metadata={k: v for k, v in metadata.items() if v is not None},
                )
            )
    return documents


def split_documents(spec: CollectionSpec, documents: list[Any]) -> list[Any]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=spec.chunk_size, chunk_overlap=spec.chunk_overlap, separators=SEPARATORS
    )
    chunks = []
    for document in documents:
        for chunk_index, chunk in enumerate(splitter.split_documents([document])):
            chunk.metadata["chunk_index"] = chunk_index
            chunk.metadata["source_id"] = source_id(
                chunk.metadata["source"], chunk.metadata["page"], chunk_index
            )
            chunks.append(chunk)
    return chunks


def ingest(spec: CollectionSpec, *, dry_run: bool = False) -> list[Any]:
    """Rebuild one collection from scratch so re-runs never leave stale chunks."""
    chunks = split_documents(spec, load_documents(spec))
    if dry_run:
        return chunks
    if not chunks:
        raise ValueError("No usable document chunks; refusing to reset the RAG collection")

    from .tools.rag import _store

    store = _store(spec.name)
    store.reset_collection()
    store.add_documents(chunks, ids=[chunk.metadata["source_id"] for chunk in chunks])
    return chunks


def _summary(chunks: list[Any]) -> str:
    rows: dict[tuple[str, str], int] = {}
    for chunk in chunks:
        key = (chunk.metadata["title"], chunk.metadata["extraction"])
        rows[key] = rows.get(key, 0) + 1
    lines = [f"{count:4d}  {extraction:<15} {title}" for (title, extraction), count in sorted(rows.items())]
    return "\n".join([*lines, f"{len(chunks):4d}  total"])


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("collection", choices=sorted(COLLECTIONS))
    parser.add_argument(
        "--dry-run", action="store_true", help="load and split only, skip embedding"
    )
    args = parser.parse_args(argv)
    chunks = ingest(COLLECTIONS[args.collection], dry_run=args.dry_run)
    print(_summary(chunks))


if __name__ == "__main__":
    main()
