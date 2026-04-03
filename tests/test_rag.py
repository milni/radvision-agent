"""Tests for Phase 3: RAG ingestion pipeline and retriever.

Strategy: a module-scoped fixture runs ingestion once into a temporary
ChromaDB directory, then all retrieval tests query that store.
This makes the tests self-contained — no dependency on a pre-existing
data/vectorstore/ directory.
"""

import pytest

from src.rag.ingest import CorpusIngestor, _split_by_h2, _extract, _first_heading
from src.rag.retriever import Retriever
from src.config import CORPUS_DIR


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vectorstore_dir(tmp_path_factory):
    """Ingest the full corpus into a fresh temporary ChromaDB store.

    Runs once per test module. Returns the path so the Retriever fixture
    can point to the same directory.
    """
    vdir = tmp_path_factory.mktemp("vectorstore")
    ingestor = CorpusIngestor(vectorstore_dir=vdir)
    counts = ingestor.ingest_all(corpus_dir=CORPUS_DIR)

    # Sanity check that ingestion actually produced chunks before tests run
    assert sum(counts.values()) > 0, "Ingestion produced no chunks"
    return vdir


@pytest.fixture(scope="module")
def retriever(vectorstore_dir):
    return Retriever(vectorstore_dir=vectorstore_dir)


# ---------------------------------------------------------------------------
# Ingestion — chunk counts
# ---------------------------------------------------------------------------


class TestIngestionCounts:
    def test_kb_articles_chunk_count(self, vectorstore_dir):
        """One chunk per KB article — 6 articles = 6 chunks."""
        ingestor = CorpusIngestor(vectorstore_dir=vectorstore_dir)
        col = ingestor._client.get_collection("kb_articles")
        assert col.count() == 6

    def test_product_docs_chunk_count(self, vectorstore_dir):
        """4 product docs, each with multiple paragraphs."""
        ingestor = CorpusIngestor(vectorstore_dir=vectorstore_dir)
        col = ingestor._client.get_collection("product_docs")
        assert col.count() >= 10

    def test_release_notes_chunk_count(self, vectorstore_dir):
        """4 versions × 2 sections (What's New + Known Issues) = 8 chunks."""
        ingestor = CorpusIngestor(vectorstore_dir=vectorstore_dir)
        col = ingestor._client.get_collection("release_notes")
        assert col.count() == 8

    def test_past_tickets_chunk_count(self, vectorstore_dir):
        """One chunk per ticket — 5 tickets = 5 chunks."""
        ingestor = CorpusIngestor(vectorstore_dir=vectorstore_dir)
        col = ingestor._client.get_collection("past_tickets")
        assert col.count() == 5


# ---------------------------------------------------------------------------
# Retrieval — result structure
# ---------------------------------------------------------------------------


class TestRetrievalStructure:
    def test_result_has_required_keys(self, retriever: Retriever):
        results = retriever.retrieve("DICOM error", ["kb_articles"], top_k=1)
        assert len(results) == 1
        r = results[0]
        assert "text" in r
        assert "metadata" in r
        assert "score" in r
        assert "source_collection" in r

    def test_score_between_zero_and_one(self, retriever: Retriever):
        results = retriever.retrieve("rendering GPU memory", ["kb_articles"], top_k=3)
        for r in results:
            assert 0.0 <= r["score"] <= 1.0

    def test_results_sorted_by_score_descending(self, retriever: Retriever):
        results = retriever.retrieve("FHIR connection pool", ["kb_articles"], top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_source_collection_matches_queried(self, retriever: Retriever):
        results = retriever.retrieve("DICOM TLS", ["past_tickets"], top_k=3)
        for r in results:
            assert r["source_collection"] == "past_tickets"

    def test_metadata_has_source_type(self, retriever: Retriever):
        results = retriever.retrieve("cardiac rendering", ["kb_articles"], top_k=1)
        assert results[0]["metadata"]["source_type"] == "kb_article"

    def test_top_k_limits_results(self, retriever: Retriever):
        results = retriever.retrieve("error", ["kb_articles", "past_tickets"], top_k=3)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# Retrieval — semantic relevance
# ---------------------------------------------------------------------------


class TestRetrievalRelevance:
    def test_tls_dicom_query_returns_kb4231(self, retriever: Retriever):
        """A query about DICOM TLS errors should surface KB-4231."""
        results = retriever.retrieve(
            "DICOM association rejected TLS renegotiation error",
            ["kb_articles"],
            top_k=3,
        )
        kb_ids = [r["metadata"].get("kb_id") for r in results]
        assert "KB-4231" in kb_ids

    def test_cardiac_rendering_query_returns_kb4298(self, retriever: Retriever):
        """A query about cardiac 4D rendering artifacts should surface KB-4298."""
        results = retriever.retrieve(
            "cardiac 4D rendering artifacts GPU VRAM",
            ["kb_articles"],
            top_k=3,
        )
        kb_ids = [r["metadata"].get("kb_id") for r in results]
        assert "KB-4298" in kb_ids

    def test_fhir_503_query_returns_kb4315(self, retriever: Retriever):
        """A query about FHIR 503 errors should surface KB-4315."""
        results = retriever.retrieve(
            "FHIR endpoint returns 503 service unavailable",
            ["kb_articles"],
            top_k=3,
        )
        kb_ids = [r["metadata"].get("kb_id") for r in results]
        assert "KB-4315" in kb_ids

    def test_screen_sharing_latency_returns_kb4330(self, retriever: Retriever):
        """A query about screen sharing lag should surface KB-4330."""
        results = retriever.retrieve(
            "screen sharing latency WebRTC collaboration",
            ["kb_articles"],
            top_k=3,
        )
        kb_ids = [r["metadata"].get("kb_id") for r in results]
        assert "KB-4330" in kb_ids

    def test_dicom_query_returns_ticket_tkt0891(self, retriever: Retriever):
        """A query about DICOM failures should surface the matching past ticket."""
        results = retriever.retrieve(
            "DICOM association failing after load balancer upgrade",
            ["past_tickets"],
            top_k=3,
        )
        ticket_ids = [r["metadata"].get("ticket_id") for r in results]
        assert "TKT-2024-0891" in ticket_ids

    def test_multi_collection_returns_both_sources(self, retriever: Retriever):
        """Querying kb_articles + past_tickets should return chunks from both."""
        results = retriever.retrieve(
            "TLS renegotiation DICOM association failure",
            ["kb_articles", "past_tickets"],
            top_k=6,
        )
        collections_seen = {r["source_collection"] for r in results}
        assert "kb_articles" in collections_seen
        assert "past_tickets" in collections_seen

    def test_product_doc_query_returns_component_config(self, retriever: Retriever):
        """A query about DICOM config settings should hit product_docs."""
        results = retriever.retrieve(
            "SCP_TLS_RENEGOTIATION configuration DICOM gateway settings",
            ["product_docs"],
            top_k=3,
        )
        assert len(results) > 0
        assert all(r["metadata"]["source_type"] == "product_doc" for r in results)

    def test_version_query_returns_release_notes(self, retriever: Retriever):
        """A query about v4.2 features should hit release_notes."""
        results = retriever.retrieve(
            "version 4.2 new features FHIR cardiac",
            ["release_notes"],
            top_k=3,
        )
        assert len(results) > 0
        assert all(r["metadata"]["source_type"] == "release_note" for r in results)


# ---------------------------------------------------------------------------
# Retrieval — edge cases
# ---------------------------------------------------------------------------


class TestRetrievalEdgeCases:
    def test_unknown_collection_returns_empty(self, retriever: Retriever):
        results = retriever.retrieve("anything", ["nonexistent_collection"], top_k=5)
        assert results == []

    def test_empty_collection_list_returns_empty(self, retriever: Retriever):
        results = retriever.retrieve("DICOM error", [], top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# Chunking utilities (unit tests, no ChromaDB needed)
# ---------------------------------------------------------------------------


class TestChunkingUtils:
    def test_split_by_h2_extracts_sections(self):
        text = "# Title\n\nPreamble\n\n## Symptom\n\nSome symptom\n\n## Resolution\n\nSome fix"
        sections = _split_by_h2(text)
        assert len(sections) == 2
        assert sections[0] == ("Symptom", "Some symptom")
        assert sections[1] == ("Resolution", "Some fix")

    def test_split_by_h2_skips_empty_sections(self):
        text = "# Title\n\n## Empty\n\n## HasContent\n\nContent here"
        sections = _split_by_h2(text)
        assert len(sections) == 1
        assert sections[0][0] == "HasContent"

    def test_extract_finds_pattern(self):
        text = "**Affected Version:** 4.2\nother content"
        assert _extract(text, r"\*\*Affected Version:\*\* (.+)") == "4.2"

    def test_extract_returns_unknown_on_no_match(self):
        assert _extract("no match here", r"\*\*Version:\*\* (.+)") == "unknown"

    def test_first_heading_extracts_title(self):
        text = "# KB-4231: Some Title\n\nContent"
        assert _first_heading(text) == "KB-4231: Some Title"

    def test_first_heading_returns_empty_on_no_heading(self):
        assert _first_heading("no headings here") == ""
