"""Integration tests for the assembled LangGraph agent (Phase 6).

Two categories:
  1. Routing unit tests — exercise the three conditional edge functions
     directly. No ChromaDB or MLflow needed.
  2. End-to-end tests — run full queries through the compiled graph.
     Automatically skipped when the vectorstore hasn't been populated yet.
     Run `python scripts/ingest_corpus.py` to enable them.
"""

import pytest

from src.agents.graph import (
    _route_from_grounding,
    _route_from_sufficiency,
    _route_from_triage,
    build_graph,
    run_agent,
)
from src.config import GROUNDING_MAX_REGENERATIONS, VECTORSTORE_DIR

# ---------------------------------------------------------------------------
# Skip marker for tests that need a populated vectorstore
# ---------------------------------------------------------------------------

_vectorstore_ready = (
    VECTORSTORE_DIR.exists()
    and any(VECTORSTORE_DIR.iterdir())
)
_needs_vectorstore = pytest.mark.skipif(
    not _vectorstore_ready,
    reason="Vectorstore not populated — run scripts/ingest_corpus.py first",
)


# ---------------------------------------------------------------------------
# Routing — triage → tool/subgraph
# ---------------------------------------------------------------------------


class TestRouteFromTriage:
    def test_error_pattern_routes_to_log_analyzer(self):
        assert _route_from_triage({"route_decision": "error_pattern"}) == "run_log_analyzer"

    def test_config_check_routes_to_compat_checker(self):
        assert _route_from_triage({"route_decision": "config_check"}) == "run_compat_checker"

    def test_docs_kb_routes_to_rag(self):
        assert _route_from_triage({"route_decision": "docs_kb"}) == "run_rag"

    def test_unknown_route_defaults_to_rag(self):
        assert _route_from_triage({"route_decision": "unexpected"}) == "run_rag"

    def test_missing_route_defaults_to_rag(self):
        assert _route_from_triage({}) == "run_rag"


# ---------------------------------------------------------------------------
# Routing — sufficiency → synthesizer | fallback tool
# ---------------------------------------------------------------------------


class TestRouteFromSufficiency:
    def test_sufficient_routes_to_synthesizer(self):
        assert _route_from_sufficiency({"evidence_sufficient": True}) == "synthesizer"

    def test_active_retry_from_docs_kb_goes_to_log_analyzer(self):
        state = {
            "evidence_sufficient": False,
            "retry_reason": "No matching evidence via 'docs_kb'. Retrying with 'error_pattern'.",
            "route_decision": "docs_kb",
        }
        assert _route_from_sufficiency(state) == "run_log_analyzer"

    def test_active_retry_from_error_pattern_goes_to_rag(self):
        state = {
            "evidence_sufficient": False,
            "retry_reason": "No matching evidence via 'error_pattern'. Retrying with 'docs_kb'.",
            "route_decision": "error_pattern",
        }
        assert _route_from_sufficiency(state) == "run_rag"

    def test_active_retry_from_config_check_goes_to_rag(self):
        state = {
            "evidence_sufficient": False,
            "retry_reason": "No matching evidence via 'config_check'. Retrying with 'docs_kb'.",
            "route_decision": "config_check",
        }
        assert _route_from_sufficiency(state) == "run_rag"

    def test_max_retries_exhausted_routes_to_synthesizer(self):
        state = {
            "evidence_sufficient": False,
            "retry_reason": "Evidence insufficient after max retries. Proceeding.",
            "route_decision": "docs_kb",
        }
        assert _route_from_sufficiency(state) == "synthesizer"

    def test_no_retry_reason_routes_to_synthesizer(self):
        assert _route_from_sufficiency({"evidence_sufficient": False, "retry_reason": ""}) == "synthesizer"


# ---------------------------------------------------------------------------
# Routing — grounding → escalation | increment_regen
# ---------------------------------------------------------------------------


class TestRouteFromGrounding:
    def test_pass_routes_to_escalation(self):
        assert _route_from_grounding({"grounding_pass": True, "grounding_regen_count": 0}) == "escalation"

    def test_fail_with_regen_slot_routes_to_increment(self):
        assert _route_from_grounding({"grounding_pass": False, "grounding_regen_count": 0}) == "increment_regen"

    def test_fail_at_max_regen_routes_to_escalation(self):
        state = {"grounding_pass": False, "grounding_regen_count": GROUNDING_MAX_REGENERATIONS}
        assert _route_from_grounding(state) == "escalation"

    def test_pass_at_max_regen_still_routes_to_escalation(self):
        # Even when regen count is at max, a passing grounding goes to escalation (correct)
        state = {"grounding_pass": True, "grounding_regen_count": GROUNDING_MAX_REGENERATIONS}
        assert _route_from_grounding(state) == "escalation"

    def test_missing_regen_count_defaults_to_zero(self):
        # No grounding_regen_count key → treated as 0 → regen slot available
        assert _route_from_grounding({"grounding_pass": False}) == "increment_regen"


# ---------------------------------------------------------------------------
# End-to-end — full graph with real ChromaDB
# ---------------------------------------------------------------------------


@_needs_vectorstore
class TestRunAgent:
    def test_returns_valid_outcome_and_response(self):
        result = run_agent(
            "DICOM association rejected with TLS renegotiation error on v4.2", "support"
        )
        assert result.get("outcome") in ("resolved", "clarify", "escalate")
        assert isinstance(result.get("final_response"), str)
        assert len(result["final_response"]) > 0

    def test_vague_query_returns_valid_outcome(self):
        # "something is not working" may clarify, escalate, or even best-effort resolve
        # if RAG finds a result above the relevance threshold — all are valid.
        result = run_agent("something is not working", "support")
        assert result["outcome"] in ("resolved", "clarify", "escalate")
        assert len(result.get("final_response", "")) > 0

    def test_config_check_route_taken_for_compat_query(self):
        result = run_agent("Is NVIDIA Quadro supported in version 4.2", "support")
        assert result.get("route_decision") == "config_check"
        assert result["outcome"] in ("resolved", "escalate", "clarify")

    def test_all_personas_return_non_empty_response(self):
        for persona in ("support", "field_engineer", "sales"):
            result = run_agent("DICOM TLS error on v4.2", persona)
            assert result["outcome"] in ("resolved", "clarify", "escalate")
            assert len(result.get("final_response", "")) > 0

    def test_safety_query_always_escalates(self):
        result = run_agent("HIPAA compliance issue with patient data", "support")
        assert result["outcome"] == "escalate"
