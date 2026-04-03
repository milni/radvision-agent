"""Tests for Phase 4: RAG subgraph nodes and end-to-end subgraph invocation.

Unit tests (TestQueryRewriter, TestIndexSelector, TestReranker, TestRelevanceGate)
do not touch ChromaDB. Integration tests (TestRAGSubgraphIntegration) ingest the
corpus into a temporary store and invoke the compiled subgraph end-to-end.
"""

import pytest

from src.config import CORPUS_DIR
from src.rag.ingest import CorpusIngestor
from src.rag.reranker import rerank
from src.rag.rewriter import rewrite_query
from src.rag.subgraph import build_rag_subgraph, relevance_gate, select_indexes


# ---------------------------------------------------------------------------
# Shared fixture — one ingested vectorstore for all integration tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vectorstore_dir(tmp_path_factory):
    vdir = tmp_path_factory.mktemp("vectorstore")
    ingestor = CorpusIngestor(vectorstore_dir=vdir)
    ingestor.ingest_all(corpus_dir=CORPUS_DIR)
    return vdir


@pytest.fixture(scope="module")
def rag_subgraph(vectorstore_dir):
    return build_rag_subgraph(vectorstore_dir=vectorstore_dir)


# ---------------------------------------------------------------------------
# Query Rewriter — unit tests
# ---------------------------------------------------------------------------


class TestQueryRewriter:
    def test_dicom_query_expands(self):
        state = {"query": "DICOM association rejected", "rewrite_count": 0}
        result = rewrite_query(state)
        expanded = result["rewritten_query"]
        assert "DIMSE" in expanded or "SCP" in expanded

    def test_tls_query_expands(self):
        state = {"query": "TLS renegotiation error", "rewrite_count": 0}
        result = rewrite_query(state)
        assert "certificate" in result["rewritten_query"] or "SSL" in result["rewritten_query"]

    def test_original_query_preserved(self):
        state = {"query": "DICOM timeout", "rewrite_count": 0}
        result = rewrite_query(state)
        assert result["rewritten_query"].startswith("DICOM timeout")

    def test_no_duplicate_terms(self):
        state = {"query": "DICOM SCP association", "rewrite_count": 0}
        result = rewrite_query(state)
        words = result["rewritten_query"].lower().split()
        assert len(words) == len(set(words)), "Duplicate terms in rewritten query"

    def test_unknown_query_unchanged(self):
        state = {"query": "completely unrelated xyz query", "rewrite_count": 0}
        result = rewrite_query(state)
        assert result["rewritten_query"] == "completely unrelated xyz query"

    def test_retry_adds_fallback_terms(self):
        state = {"query": "some query", "rewrite_count": 1}
        result = rewrite_query(state)
        # On retry, fallback terms should be added
        assert result["rewritten_query"] != "some query"

    def test_returns_rewritten_query_key(self):
        result = rewrite_query({"query": "anything", "rewrite_count": 0})
        assert "rewritten_query" in result


# ---------------------------------------------------------------------------
# Index Selector — unit tests
# ---------------------------------------------------------------------------


class TestIndexSelector:
    def test_troubleshooting_query_selects_kb_and_tickets(self):
        state = {"query": "DICOM error timeout", "rewritten_query": "DICOM error timeout", "rewrite_count": 0}
        result = select_indexes(state)
        assert "kb_articles" in result["collections"]
        assert "past_tickets" in result["collections"]

    def test_version_query_selects_release_notes(self):
        state = {"query": "what's new in v4.2", "rewritten_query": "what's new in v4.2", "rewrite_count": 0}
        result = select_indexes(state)
        assert result["collections"] == ["release_notes"]

    def test_config_query_selects_product_docs(self):
        state = {"query": "how to configure the DICOM gateway setting", "rewritten_query": "how to configure the DICOM gateway setting", "rewrite_count": 0}
        result = select_indexes(state)
        assert result["collections"] == ["product_docs"]

    def test_unclear_query_selects_all_collections(self):
        state = {"query": "tell me something", "rewritten_query": "tell me something", "rewrite_count": 0}
        result = select_indexes(state)
        assert len(result["collections"]) == 4

    def test_retry_always_selects_all_collections(self):
        # Even a version query should widen on retry
        state = {"query": "v4.2 release", "rewritten_query": "v4.2 release", "rewrite_count": 1}
        result = select_indexes(state)
        assert len(result["collections"]) == 4

    def test_returns_collections_key(self):
        result = select_indexes({"query": "anything", "rewritten_query": "anything", "rewrite_count": 0})
        assert "collections" in result


# ---------------------------------------------------------------------------
# Reranker — unit tests
# ---------------------------------------------------------------------------


def _make_result(collection, source_file, score, version="unknown", text="chunk text"):
    return {
        "text": text,
        "score": score,
        "source_collection": collection,
        "metadata": {"source_file": source_file, "version": version, "source_type": collection},
    }


class TestReranker:
    def test_deduplicates_same_source_file(self):
        results = [
            _make_result("kb_articles", "KB-4231.md", 0.8),
            _make_result("kb_articles", "KB-4231.md", 0.6),  # duplicate, lower score
            _make_result("kb_articles", "KB-4298.md", 0.7),
        ]
        state = {"raw_results": results, "rewritten_query": "DICOM error", "top_k": 5}
        out = rerank(state)
        files = [r["metadata"]["source_file"] for r in out["rag_results"]]
        assert files.count("KB-4231.md") == 1

    def test_keeps_higher_score_on_dedup(self):
        results = [
            _make_result("kb_articles", "KB-4231.md", 0.6),
            _make_result("kb_articles", "KB-4231.md", 0.8),
        ]
        state = {"raw_results": results, "rewritten_query": "DICOM error", "top_k": 5}
        out = rerank(state)
        assert out["rag_results"][0]["score"] >= 0.8

    def test_sorted_by_score_descending(self):
        results = [
            _make_result("kb_articles", "A.md", 0.5),
            _make_result("kb_articles", "B.md", 0.9),
            _make_result("kb_articles", "C.md", 0.7),
        ]
        state = {"raw_results": results, "rewritten_query": "query", "top_k": 5}
        out = rerank(state)
        scores = [r["score"] for r in out["rag_results"]]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_output(self):
        results = [_make_result("kb_articles", f"{i}.md", 0.5) for i in range(10)]
        state = {"raw_results": results, "rewritten_query": "query", "top_k": 3}
        out = rerank(state)
        assert len(out["rag_results"]) <= 3

    def test_version_boost_applied(self):
        results = [
            _make_result("release_notes", "v4.2.md", 0.6, version="4.2"),
            _make_result("release_notes", "v4.1.md", 0.65, version="4.1"),
        ]
        state = {"raw_results": results, "rewritten_query": "version 4.2 features", "top_k": 5}
        out = rerank(state)
        # v4.2 should rank first after boost
        assert out["rag_results"][0]["metadata"]["version"] == "4.2"

    def test_empty_raw_results(self):
        state = {"raw_results": [], "rewritten_query": "query", "top_k": 5}
        out = rerank(state)
        assert out["rag_results"] == []


# ---------------------------------------------------------------------------
# Relevance Gate — unit tests
# ---------------------------------------------------------------------------


class TestRelevanceGate:
    def _state(self, top_score, rewrite_count=0):
        results = [{"score": top_score, "text": "x", "metadata": {}, "source_collection": "kb_articles"}] if top_score > 0 else []
        return {"rag_results": results, "rewrite_count": rewrite_count}

    def test_high_score_passes(self):
        assert relevance_gate(self._state(0.8)) == "pass"

    def test_score_at_threshold_passes(self):
        from src.config import RAG_RELEVANCE_THRESHOLD
        assert relevance_gate(self._state(RAG_RELEVANCE_THRESHOLD)) == "pass"

    def test_low_score_retries(self):
        assert relevance_gate(self._state(0.1, rewrite_count=0)) == "retry"

    def test_low_score_passes_after_max_retries(self):
        from src.config import RAG_MAX_RETRIES
        assert relevance_gate(self._state(0.1, rewrite_count=RAG_MAX_RETRIES)) == "pass"

    def test_empty_results_retries(self):
        assert relevance_gate({"rag_results": [], "rewrite_count": 0}) == "retry"

    def test_empty_results_passes_after_max_retries(self):
        from src.config import RAG_MAX_RETRIES
        assert relevance_gate({"rag_results": [], "rewrite_count": RAG_MAX_RETRIES}) == "pass"


# ---------------------------------------------------------------------------
# RAG Subgraph — integration tests
# ---------------------------------------------------------------------------


class TestRAGSubgraphIntegration:
    def test_returns_rag_results(self, rag_subgraph):
        out = rag_subgraph.invoke({"query": "DICOM error", "top_k": 3, "rewrite_count": 0})
        assert "rag_results" in out
        assert isinstance(out["rag_results"], list)

    def test_result_structure(self, rag_subgraph):
        out = rag_subgraph.invoke({"query": "DICOM TLS error", "top_k": 3, "rewrite_count": 0})
        for r in out["rag_results"]:
            assert "text" in r
            assert "metadata" in r
            assert "score" in r
            assert "source_collection" in r

    def test_tls_dicom_query_returns_kb4231(self, rag_subgraph):
        out = rag_subgraph.invoke({
            "query": "DICOM association rejected TLS renegotiation error",
            "top_k": 5,
            "rewrite_count": 0,
        })
        kb_ids = [r["metadata"].get("kb_id") for r in out["rag_results"]]
        assert "KB-4231" in kb_ids

    def test_version_query_hits_release_notes(self, rag_subgraph):
        out = rag_subgraph.invoke({
            "query": "what's new in version 4.2",
            "top_k": 3,
            "rewrite_count": 0,
        })
        source_types = {r["metadata"]["source_type"] for r in out["rag_results"]}
        assert "release_note" in source_types

    def test_config_query_hits_product_docs(self, rag_subgraph):
        out = rag_subgraph.invoke({
            "query": "how to configure SCP_TLS_RENEGOTIATION setting",
            "top_k": 3,
            "rewrite_count": 0,
        })
        source_types = {r["metadata"]["source_type"] for r in out["rag_results"]}
        assert "product_doc" in source_types

    def test_results_sorted_by_score(self, rag_subgraph):
        out = rag_subgraph.invoke({"query": "FHIR 503 error", "top_k": 5, "rewrite_count": 0})
        scores = [r["score"] for r in out["rag_results"]]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self, rag_subgraph):
        out = rag_subgraph.invoke({"query": "error", "top_k": 2, "rewrite_count": 0})
        assert len(out["rag_results"]) <= 2

    def test_rewritten_query_present(self, rag_subgraph):
        out = rag_subgraph.invoke({"query": "DICOM fail", "top_k": 3, "rewrite_count": 0})
        assert "rewritten_query" in out
        # Rewriter should have expanded it
        assert len(out["rewritten_query"]) >= len("DICOM fail")
