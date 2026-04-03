"""Cross-index Reranker node for the RAG subgraph.

Takes raw retrieval results from one or more collections, deduplicates by
source file, applies a small version-match boost, re-sorts by score, and
returns the top_k results.

Deduplication key: (source_collection, source_file) — if the same chunk
appears via multiple collection queries, keep only the highest-scoring copy.
"""

import logging
import re

from src.config import RAG_TOP_K

logger = logging.getLogger(__name__)

_VERSION_BOOST = 0.05  # Added when chunk version matches query version


def rerank(state: dict) -> dict:
    """Deduplicate, boost, and trim retrieval results.

    Reads:
        state['raw_results']      list[dict] from the Retriever node
        state['rewritten_query']  used to extract version for boosting
        state['top_k']            how many results to return (default RAG_TOP_K)

    Returns:
        {"rag_results": list[dict]}  top_k results, sorted by score descending
    """
    results: list[dict] = state.get("raw_results", [])
    query: str = state.get("rewritten_query", state.get("query", ""))
    top_k: int = state.get("top_k", RAG_TOP_K)

    # Deduplicate: keep highest-scoring copy of each (collection, source_file) pair
    seen: dict[tuple, dict] = {}
    for r in results:
        key = (r["source_collection"], r["metadata"].get("source_file", r["text"][:80]))
        if key not in seen or r["score"] > seen[key]["score"]:
            seen[key] = r

    deduped = list(seen.values())

    # Version boost: if the query mentions a specific version, reward matching chunks
    version_match = re.search(r"\bv?(\d+\.\d+(?:\.\d+)?)\b", query)
    if version_match:
        target = version_match.group(1)
        for r in deduped:
            if r["metadata"].get("version", "") == target:
                r = dict(r)  # copy to avoid mutating the original
                r["score"] = min(1.0, r["score"] + _VERSION_BOOST)
                seen[(r["source_collection"], r["metadata"].get("source_file", ""))] = r

        deduped = list(seen.values())

    deduped.sort(key=lambda r: r["score"], reverse=True)
    rag_results = deduped[:top_k]

    logger.debug(
        "Reranker: %d raw -> %d deduped -> %d returned",
        len(results), len(deduped), len(rag_results),
    )
    return {"rag_results": rag_results}
