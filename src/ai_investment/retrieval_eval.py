"""Retrieval evaluation following design section B-3.

    # 1) (question, answer chunk) ground truth: 20 Korean + 10 English chunks
    PYTHONPATH=src python -m ai_investment.retrieval_eval questions market
    # 2) Recall@5 / MRR@5 per retriever and embedding model
    PYTHONPATH=src python -m ai_investment.retrieval_eval score market --models kure bge-m3 e5
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .ingest import COLLECTIONS, CollectionSpec, load_documents, split_documents
from .tools.rag import hybrid_search, mmr_search


K = 5
MIN_CHUNK_CHARS = 200
HANGUL = re.compile(r"[가-힣]")
LATIN = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class EmbeddingModel:
    name: str
    query_prefix: str = ""
    passage_prefix: str = ""


MODELS = {
    "kure": EmbeddingModel("nlpai-lab/KURE-v1"),
    "bge-m3": EmbeddingModel("BAAI/bge-m3"),
    "e5": EmbeddingModel("intfloat/multilingual-e5-large", "query: ", "passage: "),
}

QUALITATIVE_QUESTIONS = [
    "휴머노이드 시장 성장률은?",
    "2024년 전 세계 산업용 로봇 설치 대수는?",
    "한국의 제조업 로봇 밀도는?",
    "국내 물류 자율주행 로봇 스타트업은 어디가 있나?",
    "서비스 로봇 중 가장 많이 설치된 용도는?",
]


def chunk_language(text: str) -> str:
    """'ko' if Hangul dominates letters, else 'en'."""
    hangul, latin = len(HANGUL.findall(text)), len(LATIN.findall(text))
    return "ko" if hangul >= latin else "en"


def sample_chunks(
    chunks: Sequence[Any], *, n_ko: int = 20, n_en: int = 10, seed: int = 42
) -> list[Any]:
    rng = random.Random(seed)
    pools: dict[str, list[Any]] = {"ko": [], "en": []}
    for chunk in chunks:
        if len(chunk.page_content) >= MIN_CHUNK_CHARS:
            pools[chunk_language(chunk.page_content)].append(chunk)
    for lang, n in (("ko", n_ko), ("en", n_en)):
        if len(pools[lang]) < n:
            raise ValueError(f"only {len(pools[lang])} {lang} chunks, need {n}")
    return rng.sample(pools["ko"], n_ko) + rng.sample(pools["en"], n_en)


class GeneratedQuestion(BaseModel):
    question: str = Field(description="실제 사용자가 검색창에 칠 법한 50자 이내 한국어 질문")


def generate_questions(chunks: Sequence[Any], llm) -> list[dict[str, Any]]:
    structured = llm.with_structured_output(GeneratedQuestion)
    items = []
    for chunk in chunks:
        result = structured.invoke(
            "다음 문서 청크가 정답 근거가 되는 한국어 질문을 하나 만드세요.\n"
            "- 투자 심사역이 검색창에 칠 법한 50자 이내의 짧은 질문 한 문장\n"
            "- 답(수치·기업명)은 질문에 넣지 않고, 청크 문장을 베끼지 않습니다\n"
            "- 문서 제목이나 '이 문서', '이 리포트' 같은 표현을 쓰지 않습니다\n\n"
            f"{chunk.page_content}"
        )
        items.append(
            {
                "question": result.question.strip(),
                "chunk_id": chunk.metadata["source_id"],
                "lang": chunk_language(chunk.page_content),
                "title": chunk.metadata["title"],
                "page": chunk.metadata["page"] + 1,
            }
        )
    return items


def recall_mrr(ranked: Sequence[Sequence[str]], answers: Sequence[str], k: int = K) -> tuple[float, float]:
    if not answers:
        raise ValueError("no questions to score")
    hits, reciprocal = 0, 0.0
    for ids, answer in zip(ranked, answers, strict=True):
        top = list(ids)[:k]
        if answer in top:
            hits += 1
            reciprocal += 1 / (top.index(answer) + 1)
    return hits / len(answers), reciprocal / len(answers)


class SentenceEmbeddings:
    def __init__(self, model: EmbeddingModel) -> None:
        from sentence_transformers import SentenceTransformer

        self.spec = model
        self.model = SentenceTransformer(model.name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [self.spec.passage_prefix + text for text in texts]
        return self.model.encode(prefixed, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(self.spec.query_prefix + text, normalize_embeddings=True).tolist()


def _ephemeral_store(chunks: Sequence[Any], embeddings, name: str):
    import chromadb
    from langchain_chroma import Chroma

    store = Chroma(
        collection_name=name,
        embedding_function=embeddings,
        client=chromadb.EphemeralClient(),
    )
    store.reset_collection()
    store.add_documents(list(chunks), ids=[c.metadata["source_id"] for c in chunks])
    return store


def _bm25(chunks: Sequence[Any]) -> Callable[[str], list[Any]]:
    from langchain_community.retrievers import BM25Retriever

    retriever = BM25Retriever.from_documents(list(chunks))
    retriever.k = K
    return retriever.invoke


def score(
    spec: CollectionSpec, questions: Sequence[dict[str, Any]], model_keys: Sequence[str]
) -> list[dict[str, Any]]:
    chunks = split_documents(spec, load_documents(spec))
    known = {c.metadata["source_id"] for c in chunks}
    stale = [q["chunk_id"] for q in questions if q["chunk_id"] not in known]
    if stale:
        raise ValueError(f"{len(stale)} answer chunks no longer exist; regenerate questions")

    retrievers: dict[str, Callable[[str], list[Any]]] = {"bm25": _bm25(chunks)}
    for key in model_keys:
        store = _ephemeral_store(chunks, SentenceEmbeddings(MODELS[key]), f"eval_{key}")
        retrievers[f"{key}/dense"] = lambda q, s=store: s.similarity_search(q, k=K)
        retrievers[f"{key}/mmr"] = lambda q, s=store: mmr_search(s, q, K)
        retrievers[f"{key}/hybrid"] = lambda q, s=store: hybrid_search(s, q, K)

    rows = []
    for name, retrieve in retrievers.items():
        ranked = [[d.metadata["source_id"] for d in retrieve(q["question"])] for q in questions]
        row: dict[str, Any] = {"retriever": name}
        for group in ("ko", "en", "all"):
            idx = [i for i, q in enumerate(questions) if group == "all" or q["lang"] == group]
            recall, mrr = recall_mrr([ranked[i] for i in idx], [questions[i]["chunk_id"] for i in idx])
            row[group] = {"recall@5": round(recall, 3), "mrr@5": round(mrr, 3), "n": len(idx)}
        rows.append(row)
    return rows


def qualitative(spec: CollectionSpec) -> list[dict[str, Any]]:
    """Top-5 of the production market search for team review."""
    from .tools.rag import search_market_docs, search_tech_docs

    search = search_market_docs if spec.name == COLLECTIONS["market"].name else search_tech_docs
    return [
        {
            "question": question,
            "results": [
                {"title": e.title, "page": e.page, "excerpt": e.excerpt[:160].replace("\n", " ")}
                for e in search(question, k=K)
            ],
        }
        for question in QUALITATIVE_QUESTIONS
    ]


def _questions_path(spec: CollectionSpec) -> Path:
    return spec.root / "eval" / "questions.json"


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="retrieval evaluation (design B-3)")
    parser.add_argument("command", choices=["questions", "score", "qualitative"])
    parser.add_argument("collection", choices=sorted(COLLECTIONS))
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), default=["kure"])
    args = parser.parse_args(argv)
    spec = COLLECTIONS[args.collection]
    path = _questions_path(spec)

    if args.command == "questions":
        from .agents.market import _default_llm

        chunks = sample_chunks(split_documents(spec, load_documents(spec)))
        path.parent.mkdir(parents=True, exist_ok=True)
        items = generate_questions(chunks, _default_llm())
        path.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {len(items)} questions to {path}")
    elif args.command == "score":
        questions = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps(score(spec, questions, args.models), ensure_ascii=False, indent=1))
    else:
        print(json.dumps(qualitative(spec), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
