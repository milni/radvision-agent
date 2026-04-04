"""RAG subgraph for the RadVision Pro support agent.

Nodes (in order):
  rewrite         — Query Rewriter: domain synonym expansion
  select          — Index Selector: choose which ChromaDB collections to query
  retrieve        — Multi-index Retriever: query selected collections
  rerank          — Cross-index Reranker: deduplicate, boost, trim to top_k
  [relevance gate]— Conditional: pass → END, low-score → increment_retry → rewrite

The subgraph is exposed via build_rag_subgraph(vectorstore_dir=None).
Pass vectorstore_dir to override the default (useful in tests).
"""

import logging
import re
from pathlib import Path
from typing import Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.config import (
    CHROMA_COLLECTION_NAMES,
    RAG_MAX_RETRIES,
    RAG_RELEVANCE_THRESHOLD,
    RAG_TOP_K,
    VECTORSTORE_DIR,
)
from src.rag.reranker import rerank
from src.rag.retriever import Retriever
from src.rag.rewriter import rewrite_query
from src.tracking import log_metrics_safe, set_tags_safe

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subgraph state
# ---------------------------------------------------------------------------


class RAGState(TypedDict):
    query: str            # Original query — never modified after entry
    rewritten_query: str  # After Query Rewriter
    collections: list[str]  # After Index Selector
    top_k: int            # How many final results to return
    raw_results: list[dict]   # After Retriever (over-fetched)
    rag_results: list[dict]   # After Reranker (final output)
    rewrite_count: int    # Rewrites attempted so far (max RAG_MAX_RETRIES)


# ---------------------------------------------------------------------------
# Index Selector — rule-based, no LLM
# ---------------------------------------------------------------------------

# Rules are evaluated in order; first match wins.
# Each entry: (regex_pattern, collections_to_query)
_SELECTOR_RULES: list[tuple[str, list[str]]] = [
    # Version / changelog signals
    (
        r"\bversion\b|\brelease\b|\bchangelog\b|\bwhat.?s new\b|\bv\d+\.\d+\b",
        ["release_notes"],
    ),
    # Configuration / setup signals
    (
        r"\bconfigur\b|\bsetting\b|\bparam(eter)?\b|\bhow.?to\b|\bsetup\b|\binstall\b",
        ["product_docs"],
    ),
    # Troubleshooting / error signals
    (
        r"\berror\b|\bfail\b|\bcrash\b|\btimeout\b|\breject\b|\bartifact\b"
        r"|\blag\b|\bslow\b|\b503\b|\bTLS\b|\bDICOM\b|\bFHIR\b",
        ["kb_articles"],
    ),
]


def select_indexes(state: dict) -> dict:
    """Choose which collections to search based on the rewritten query.

    On a retry (rewrite_count > 0) always searches all three collections to
    widen the net. Otherwise the first matching rule wins; if no rule matches,
    all three collections are searched.
    """
    query = state.get("rewritten_query", state["query"])
    rewrite_count = state.get("rewrite_count", 0)

    if rewrite_count > 0:
        selected = list(CHROMA_COLLECTION_NAMES)
        logger.debug("Index selector (retry): all %d collections", len(selected))
        return {"collections": selected}

    for pattern, cols in _SELECTOR_RULES:
        if re.search(pattern, query, re.IGNORECASE):
            logger.debug("Index selector: %s (matched %r)", cols, pattern)
            return {"collections": cols}

    logger.debug("Index selector: all collections (no rule matched)")
    return {"collections": list(CHROMA_COLLECTION_NAMES)}


# ---------------------------------------------------------------------------
# Relevance Gate — conditional edge router
# ---------------------------------------------------------------------------


def relevance_gate(state: dict) -> Literal["pass", "retry"]:
    """Return 'pass' if the top result meets the threshold, 'retry' otherwise.

    After RAG_MAX_RETRIES rewrites, always passes to avoid infinite loops.
    """
    results = state.get("rag_results", [])
    top_score = results[0]["score"] if results else 0.0
    rewrite_count = state.get("rewrite_count", 0)

    if top_score >= RAG_RELEVANCE_THRESHOLD:
        logger.debug("Relevance gate: PASS (score=%.3f)", top_score)
        return "pass"

    if rewrite_count < RAG_MAX_RETRIES:
        logger.debug(
            "Relevance gate: RETRY (score=%.3f, rewrites=%d)", top_score, rewrite_count
        )
        return "retry"

    logger.debug(
        "Relevance gate: PASS (max retries reached, score=%.3f)", top_score
    )
    return "pass"


def _increment_rewrite_count(state: dict) -> dict:
    """Bump the rewrite counter before looping back to the rewriter."""
    return {"rewrite_count": state.get("rewrite_count", 0) + 1}


# ---------------------------------------------------------------------------
# Subgraph factory
# ---------------------------------------------------------------------------


def build_rag_subgraph(vectorstore_dir: Path | None = None):
    """Build and compile the RAG subgraph.

    Args:
        vectorstore_dir: Path to the ChromaDB directory. Defaults to
            VECTORSTORE_DIR from config. Pass a different path in tests.

    Returns:
        A compiled LangGraph that accepts RAGState-compatible input and
        returns state with rag_results populated.
    """
    vdir = vectorstore_dir or VECTORSTORE_DIR
    retriever = Retriever(vectorstore_dir=vdir)

    def retrieve(state: dict) -> dict:
        """Retrieve from selected collections, over-fetching for the reranker."""
        raw = retriever.retrieve(
            query=state["rewritten_query"],
            collections=state.get("collections", list(CHROMA_COLLECTION_NAMES)),
            top_k=state.get("top_k", RAG_TOP_K) * 3,
        )
        return {"raw_results": raw}

    def rerank_and_log(state: dict) -> dict:
        """Rerank results, then log RAG metrics into the active MLflow run."""
        result = rerank(state)
        rag_results = result.get("rag_results", [])
        top_score = rag_results[0]["score"] if rag_results else 0.0
        collections = state.get("collections", [])
        log_metrics_safe({
            "rag.top_score": round(top_score, 4),
            "rag.num_results": len(rag_results),
            "rag.rewrite_count": state.get("rewrite_count", 0),
            "rag.collections_count": len(collections),
        })
        set_tags_safe({"rag.collections": ",".join(collections)})
        return result

    graph = StateGraph(RAGState)
    graph.add_node("rewrite", rewrite_query)
    graph.add_node("select", select_indexes)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank_and_log)
    graph.add_node("increment_retry", _increment_rewrite_count)

    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "select")
    graph.add_edge("select", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_conditional_edges(
        "rerank",
        relevance_gate,
        {"pass": END, "retry": "increment_retry"},
    )
    graph.add_edge("increment_retry", "rewrite")

    return graph.compile()
