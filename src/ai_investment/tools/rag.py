from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path

from ..models import Evidence
from .common import documents_to_evidence


MODEL_NAME = "nlpai-lab/KURE-v1"
TECH_COLLECTION = "tech_docs"
MARKET_COLLECTION = "market_docs"
DEFAULT_CHROMA_PATH = "data/chroma"
# Parallel agents (technical, market) and CRAG topics share these singletons.
# Concurrent first-time construction crashes Chroma's client, and concurrent
# SentenceTransformer.encode calls segfault torch, so both are serialized.
_INIT_LOCK = threading.Lock()
_ENCODE_LOCK = threading.Lock()


class KUREEmbeddings:
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(MODEL_NAME)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with _ENCODE_LOCK:
            return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        with _ENCODE_LOCK:
            return self.model.encode(text, normalize_embeddings=True).tolist()


def _embeddings() -> KUREEmbeddings:
    with _INIT_LOCK:
        return _cached_embeddings()


def _store(collection_name: str):
    with _INIT_LOCK:
        return _cached_store(collection_name)


@lru_cache(maxsize=1)
def _cached_embeddings() -> KUREEmbeddings:
    return KUREEmbeddings()


@lru_cache(maxsize=2)
def _cached_store(collection_name: str):
    from langchain_chroma import Chroma

    persist_directory = Path(
        os.getenv("CHROMA_PERSIST_DIRECTORY", DEFAULT_CHROMA_PATH)
    )
    return Chroma(
        collection_name=collection_name,
        embedding_function=_cached_embeddings(),
        persist_directory=str(persist_directory),
    )


def search_tech_docs(query: str, k: int = 5) -> list[Evidence]:
    _validate_search(query, k)
    return documents_to_evidence(mmr_search(_store(TECH_COLLECTION), query, k))


def search_market_docs(query: str, k: int = 5) -> list[Evidence]:
    _validate_search(query, k)
    return documents_to_evidence(hybrid_search(_store(MARKET_COLLECTION), query, k))


def mmr_search(store, query: str, k: int) -> list:
    return store.max_marginal_relevance_search(query, k=k, fetch_k=20, lambda_mult=0.5)


def hybrid_search(store, query: str, k: int, weights: tuple[float, float] = (0.4, 0.6)) -> list:
    """BM25 + Dense MMR ensemble over every document in the store."""
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document

    raw = store.get(include=["documents", "metadatas"])
    documents = [
        Document(page_content=text, metadata=metadata or {})
        for text, metadata in zip(raw.get("documents", []), raw.get("metadatas", []))
    ]
    if not documents:
        return []

    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = k
    dense = store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.5},
    )
    retriever = EnsembleRetriever(retrievers=[bm25, dense], weights=list(weights))
    return retriever.invoke(query)[:k]


def _validate_search(query: str, k: int) -> None:
    if not query.strip():
        raise ValueError("query must not be empty")
    if not 1 <= k <= 5:
        raise ValueError("k must be between 1 and 5")
