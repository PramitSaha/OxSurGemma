"""
RAG retrieval: vector search → rerank → structured chunks.

All paths are relative to this folder (surgical_rag). Use from another device
by copying the whole surgical_rag folder (including data/rag_index if already built).

Input:
  {"query": "...", "top_k": 8, "filters": {"type": "procedure"}}

Output:
  {"chunks": [{"text", "book", "chapter", "page", "page_end", "score"}, ...]}
"""
from __future__ import annotations

import contextlib
import math
import os
import sys
from pathlib import Path

# Use cache in repo root (same as build_index.py) if not set
if "TRANSFORMERS_CACHE" not in os.environ and "HF_HOME" not in os.environ:
    _pipeline_dir = Path(__file__).resolve().parent
    _project_root = _pipeline_dir.parent
    cache_dir = _project_root / "hf_cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    except OSError:
        pass

# All paths relative to this folder
PIPELINE_DIR = Path(__file__).resolve().parent
RAG_INDEX_PATH = PIPELINE_DIR / "data" / "rag_index"
COLLECTION_NAME = "cholecystectomy_books"
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "BAAI/bge-large-en-v1.5")
RERANK_MODEL = os.environ.get("RAG_RERANK_MODEL", "BAAI/bge-reranker-base")
VECTOR_SEARCH_N = 20


def _get_cache_dir():
    return os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HOME")


@contextlib.contextmanager
def _suppress_load_report():
    """Suppress LOAD REPORT / UNEXPECTED key messages from transformers during model load."""
    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
        yield
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout, sys.stderr = old_stdout, old_stderr


def _get_embedder():
    from sentence_transformers import SentenceTransformer
    cache = _get_cache_dir()
    kwargs = {"cache_folder": cache} if cache else {}
    with _suppress_load_report():
        return SentenceTransformer(EMBED_MODEL, **kwargs)


def _get_reranker():
    from sentence_transformers import CrossEncoder
    cache = _get_cache_dir()
    kwargs = {"cache_folder": cache} if cache else {}
    with _suppress_load_report():
        return CrossEncoder(RERANK_MODEL, **kwargs)


def _get_collection():
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(
        path=str(RAG_INDEX_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(name=COLLECTION_NAME)


def _rewrite_query(query: str) -> str:
    return query.strip()


def rag_retrieve(
    query: str,
    top_k: int = 8,
    filters: dict | None = None,
    *,
    rewrite_query: bool = True,
    use_reranker: bool = True,
) -> dict:
    """
    Retrieve relevant chunks: vector search → rerank → structured result.

    Returns:
        {"chunks": [{"text", "book", "chapter", "page", "page_end", "score"}, ...]}
    """
    if rewrite_query:
        query = _rewrite_query(query)
    q_embed = _get_embedder().encode([query], normalize_embeddings=False)
    coll = _get_collection()
    n = max(top_k, VECTOR_SEARCH_N) if use_reranker else top_k
    where = filters if filters else None
    res = coll.query(
        query_embeddings=q_embed.tolist(),
        n_results=n,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids = res["ids"][0]
    docs = res["documents"][0]
    metadatas = res["metadatas"][0] or []
    distances = res["distances"][0]
    if not docs:
        return {"chunks": []}
    vector_scores = [1.0 / (1.0 + d) for d in distances]
    if use_reranker and len(docs) > top_k:
        reranker = _get_reranker()
        pairs = [[query, d] for d in docs]
        rerank_scores = reranker.predict(pairs)
        indexed = list(zip(ids, docs, metadatas, rerank_scores, vector_scores))
        indexed.sort(key=lambda x: x[3], reverse=True)
        indexed = indexed[:top_k]
        chunks = []
        for id_, doc, meta, rscore, _ in indexed:
            meta = meta or {}
            score = float(1.0 / (1.0 + math.exp(-rscore))) if isinstance(rscore, (int, float)) else float(rscore)
            page_start = meta.get("page_start")
            page_end = meta.get("page_end")
            chunks.append({
                "text": doc,
                "book": meta.get("book", ""),
                "chapter": meta.get("chapter", ""),
                "page": page_start or page_end,
                "page_end": page_end,
                "score": round(score, 4),
            })
    else:
        take = min(top_k, len(docs))
        chunks = []
        for i in range(take):
            meta = metadatas[i] if i < len(metadatas) else {}
            doc = docs[i]
            score = vector_scores[i] if i < len(vector_scores) else 0.0
            page_start = meta.get("page_start")
            page_end = meta.get("page_end")
            chunks.append({
                "text": doc,
                "book": meta.get("book", ""),
                "chapter": meta.get("chapter", ""),
                "page": page_start or page_end,
                "page_end": page_end,
                "score": round(score, 4),
            })
    return {"chunks": chunks}


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="How do I safely obtain the critical view of safety?")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--filter-type", dest="filter_type", default=None, help="e.g. procedure")
    parser.add_argument("--no-rerank", action="store_true", help="skip reranker")
    args = parser.parse_args()
    filters = {"type": args.filter_type} if args.filter_type else None
    out = rag_retrieve(args.query, top_k=args.top_k, filters=filters, use_reranker=not args.no_rerank)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
