"""Tests for the five agent nodes.

Split into two groups:
  - Pure unit tests: no LLM, no ChromaDB (fast, always run)
  - Node integration tests: require Ollama (skipped when Ollama is unreachable)

All tests are marked with the group they belong to so CI can run
pure tests separately with: pytest -m "not llm"
"""

import pytest
import requests

from src.agents.grounding import grounding_node, _collect_evidence_text
from src.agents.sufficiency import sufficiency_node
from src.agents.synthesizer import synthesizer_node, _kb_refs
from src.agents.triage import triage_node
from src.agents.escalation import escalation_node
from src.config import OLLAMA_BASE_URL, SUFFICIENCY_MAX_RETRIES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rag_result(score=0.8, kb_id="KB-4231", source="kb_articles", text="some text"):
    return {
        "text": text,
        "score": score,
        "source_collection": source,
        "metadata": {"kb_id": kb_id, "source_type": "kb_article", "source_file": f"{kb_id}.md"},
    }

def _tool_match(kb_ref="KB-4231", component="DICOM Gateway"):
    return {
        "matched": True,
        "pattern_id": "DICOM-TLS-001",
        "kb_ref": kb_ref,
        "component": component,
        "severity": "high",
        "description": "TLS renegotiation disabled on SCP listener.",
        "matched_text": "TLS renegotiation",
    }

def _tool_no_match():
    return {"matched": False}

def _compat_result(status="supported"):
    return {"status": status, "reason": "GPU is on the supported list.", "recommendation": ""}


# ---------------------------------------------------------------------------
# Skip marker — requires Ollama to be running
# ---------------------------------------------------------------------------

def _ollama_available() -> bool:
    try:
        return requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2).ok
    except Exception:
        return False

_needs_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama not running — skipping LLM-dependent tests",
)


# ---------------------------------------------------------------------------
# Triage node (LLM-dependent)
# ---------------------------------------------------------------------------


@_needs_ollama
class TestTriageNode:
    def test_returns_required_keys(self):
        out = triage_node({"query": "DICOM TLS error on v4.2", "persona": "support"})
        for key in ("entities", "intent", "route_decision", "tool_results", "rag_results"):
            assert key in out

    def test_initialises_empty_results(self):
        out = triage_node({"query": "anything", "persona": "support"})
        assert out["tool_results"] == []
        assert out["rag_results"] == []

    def test_route_is_valid_value(self):
        out = triage_node({"query": "DICOM TLS renegotiation error", "persona": "support"})
        assert out["route_decision"] in ("error_pattern", "config_check", "docs_kb")

    def test_intent_is_valid_value(self):
        out = triage_node({"query": "how to configure DICOM gateway", "persona": "support"})
        assert out["intent"] in ("troubleshooting", "feature_inquiry", "config_help", "general")

    def test_no_crash_on_missing_persona(self):
        out = triage_node({"query": "DICOM error"})
        assert "intent" in out

    def test_error_log_token_routes_to_error_pattern(self):
        out = triage_node({"query": "AssocReject reason=0x0006 in DICOM gateway logs", "persona": "support"})
        assert out["route_decision"] == "error_pattern"

    def test_compat_query_routes_to_config_check(self):
        out = triage_node({"query": "Does NVIDIA T4 support vessel analysis in v4.2?", "persona": "sales"})
        assert out["route_decision"] == "config_check"

    def test_how_to_query_routes_to_docs_kb(self):
        out = triage_node({"query": "How do I configure DICOM send to an external PACS?", "persona": "support"})
        assert out["route_decision"] == "docs_kb"


# ---------------------------------------------------------------------------
# Sufficiency Check (pure unit tests — no LLM)
# ---------------------------------------------------------------------------


class TestSufficiencyNode:
    def _base_state(self, tool_results=None, rag_results=None, retry_count=0, route="docs_kb"):
        return {
            "tool_results":   tool_results or [],
            "rag_results":    rag_results or [],
            "retry_count":    retry_count,
            "route_decision": route,
        }

    def test_sufficient_with_log_match(self):
        out = sufficiency_node(self._base_state(tool_results=[_tool_match()]))
        assert out["evidence_sufficient"] is True

    def test_sufficient_with_compat_supported(self):
        out = sufficiency_node(self._base_state(tool_results=[_compat_result("supported")]))
        assert out["evidence_sufficient"] is True

    def test_sufficient_with_compat_unsupported(self):
        out = sufficiency_node(self._base_state(tool_results=[_compat_result("unsupported")]))
        assert out["evidence_sufficient"] is True

    def test_sufficient_with_compat_limited(self):
        out = sufficiency_node(self._base_state(tool_results=[_compat_result("limited")]))
        assert out["evidence_sufficient"] is True

    def test_sufficient_with_high_rag_score(self):
        out = sufficiency_node(self._base_state(rag_results=[_rag_result(score=0.8)]))
        assert out["evidence_sufficient"] is True

    def test_insufficient_with_low_rag_score(self):
        out = sufficiency_node(self._base_state(rag_results=[_rag_result(score=0.1)]))
        assert out["evidence_sufficient"] is False

    def test_insufficient_sets_retry_reason(self):
        out = sufficiency_node(self._base_state(retry_count=0))
        assert out["retry_reason"] != ""

    def test_insufficient_after_max_retries_still_proceeds(self):
        out = sufficiency_node(self._base_state(retry_count=SUFFICIENCY_MAX_RETRIES))
        assert out["evidence_sufficient"] is False
        assert "max retries" in out["retry_reason"]

    def test_no_match_tool_result_is_insufficient(self):
        out = sufficiency_node(self._base_state(tool_results=[_tool_no_match()]))
        assert out["evidence_sufficient"] is False

    def test_sufficient_sets_empty_retry_reason(self):
        out = sufficiency_node(self._base_state(tool_results=[_tool_match()]))
        assert out["retry_reason"] == ""

    def test_route_fallback_from_error_pattern_suggests_docs_kb(self):
        out = sufficiency_node(self._base_state(route="error_pattern"))
        assert "docs_kb" in out["retry_reason"]

    def test_retry_count_incremented_when_retrying(self):
        out = sufficiency_node(self._base_state(retry_count=0))
        assert out["retry_count"] == 1

    def test_retry_count_not_incremented_after_max_retries(self):
        out = sufficiency_node(self._base_state(retry_count=SUFFICIENCY_MAX_RETRIES))
        assert out["retry_count"] == SUFFICIENCY_MAX_RETRIES


# ---------------------------------------------------------------------------
# Synthesizer (LLM-dependent)
# ---------------------------------------------------------------------------


@_needs_ollama
class TestSynthesizerNode:
    def _base_state(self, persona="support", rag_results=None, tool_results=None):
        return {
            "persona":      persona,
            "tool_results": tool_results or [],
            "rag_results":  rag_results or [_rag_result(
                text="TLS renegotiation must be disabled on the SCP listener.",
                score=0.85,
            )],
            "query": "DICOM TLS error",
        }

    def test_returns_draft_response(self):
        out = synthesizer_node(self._base_state())
        assert "draft_response" in out
        assert len(out["draft_response"]) > 20

    def test_support_and_sales_give_different_responses(self):
        support_out = synthesizer_node(self._base_state(persona="support"))
        sales_out   = synthesizer_node(self._base_state(persona="sales"))
        assert support_out["draft_response"] != sales_out["draft_response"]

    def test_tool_match_takes_priority_over_rag(self):
        state = self._base_state(tool_results=[_tool_match()])
        out = synthesizer_node(state)
        # Tool match evidence contains "TLS renegotiation" — LLM should reflect it
        assert "TLS" in out["draft_response"] or "renegotiation" in out["draft_response"].lower()

    def test_no_evidence_produces_non_empty_fallback(self):
        out = synthesizer_node({"persona": "support", "tool_results": [], "rag_results": [], "query": "something"})
        assert len(out["draft_response"]) > 10


# ---------------------------------------------------------------------------
# KB refs deduplication (pure unit test — no LLM)
# ---------------------------------------------------------------------------


class TestKbRefs:
    def test_deduplication(self):
        refs = _kb_refs(
            tool_results=[{"matched": True, "kb_ref": "KB-4231", "description": "TLS"}],
            rag_results=[_rag_result(kb_id="KB-4231")],
        )
        assert refs.count("KB-4231") == 1

    def test_empty_inputs(self):
        assert _kb_refs([], []) == []

    def test_collects_from_both_sources(self):
        refs = _kb_refs(
            tool_results=[{"matched": True, "kb_ref": "KB-4231", "description": "TLS"}],
            rag_results=[_rag_result(kb_id="KB-4298")],
        )
        assert "KB-4231" in refs
        assert "KB-4298" in refs


# ---------------------------------------------------------------------------
# Grounding — collect evidence text (pure unit test)
# ---------------------------------------------------------------------------


class TestCollectEvidenceText:
    def test_tool_result_values_included(self):
        evidence = _collect_evidence_text(
            [{"kb_ref": "KB-4231", "component": "DICOM Gateway"}], []
        )
        assert "kb-4231" in evidence
        assert "dicom gateway" in evidence

    def test_rag_text_and_metadata_included(self):
        evidence = _collect_evidence_text([], [_rag_result(text="TLS fix here", kb_id="KB-4231")])
        assert "tls fix here" in evidence
        assert "kb-4231" in evidence

    def test_both_empty_returns_empty_string(self):
        assert _collect_evidence_text([], []) == ""


# ---------------------------------------------------------------------------
# Grounding node (LLM-dependent)
# ---------------------------------------------------------------------------


@_needs_ollama
class TestGroundingNode:
    def test_high_score_passes(self):
        rag = [_rag_result(text="KB-4231 TLS renegotiation fix is documented here.")]
        out = grounding_node({
            "draft_response": "See KB-4231 for the TLS fix.",
            "rag_results": rag, "tool_results": [],
        })
        assert out["grounding_pass"] is True
        assert out["grounding_score"] >= 0.5

    def test_final_response_set_from_draft(self):
        out = grounding_node({
            "draft_response": "Some response text.",
            "rag_results": [], "tool_results": [],
        })
        assert out["final_response"] == "Some response text."

    def test_empty_draft_fails_grounding(self):
        out = grounding_node({"draft_response": "", "rag_results": [], "tool_results": []})
        assert out["grounding_pass"] is False
        assert out["grounding_score"] == 0.0

    def test_whitespace_only_draft_fails_grounding(self):
        out = grounding_node({"draft_response": "   ", "rag_results": [], "tool_results": []})
        assert out["grounding_pass"] is False

    def test_no_evidence_auto_passes(self):
        # No evidence → no technical claims → trivially passes without LLM call
        out = grounding_node({
            "draft_response": "Please clarify your version and symptoms.",
            "rag_results": [], "tool_results": [],
        })
        assert out["grounding_pass"] is True
        assert out["grounding_score"] == 1.0

    def test_regen_count_passed_through_unchanged(self):
        out = grounding_node({
            "draft_response": "Some text.",
            "rag_results": [], "tool_results": [],
            "grounding_regen_count": 0,
        })
        assert out["grounding_regen_count"] == 0


# ---------------------------------------------------------------------------
# Escalation Gate (pure unit tests — no LLM)
# ---------------------------------------------------------------------------


class TestEscalationNode:
    def _base_state(self, **kwargs):
        defaults = {
            "query":              "DICOM error",
            "grounding_score":    0.9,
            "grounding_pass":     True,
            "evidence_sufficient": True,
            "entities":           {"version": "4.2", "component": "DICOM Gateway"},
            "draft_response":     "Here is the fix for the DICOM issue.",
            "tool_results":       [],
            "rag_results":        [],
        }
        defaults.update(kwargs)
        return defaults

    def test_resolved_when_grounded_and_sufficient(self):
        out = escalation_node(self._base_state())
        assert out["outcome"] == "resolved"
        assert out["final_response"] != ""

    def test_escalate_on_safety_keyword(self):
        out = escalation_node(self._base_state(query="HIPAA compliance issue with patient data"))
        assert out["outcome"] == "escalate"
        assert out["escalation_package"] is not None

    def test_escalate_on_phi_data_leak(self):
        out = escalation_node(self._base_state(query="PHI data leak through collaboration module"))
        assert out["outcome"] == "escalate"

    def test_escalate_on_critically_low_grounding(self):
        # Provide rag_results so the no-evidence shortcut is bypassed and the grounding check fires
        rag = [{"text": "some evidence", "score": 0.8, "source_collection": "kb_articles",
                "metadata": {"kb_id": "KB-4231", "source_type": "kb_article", "source_file": "KB-4231.md"}}]
        out = escalation_node(self._base_state(grounding_score=0.05, grounding_pass=False, rag_results=rag))
        assert out["outcome"] == "escalate"

    def test_no_evidence_and_no_entities_resolves_with_redirect(self):
        # When no tool_results/rag_results: the "no data found" shortcut fires
        # and returns "resolved" with a polite redirect message (not "clarify").
        out = escalation_node(self._base_state(
            evidence_sufficient=False, entities={}, grounding_pass=False, grounding_score=0.5,
        ))
        assert out["outcome"] == "resolved"
        assert len(out["final_response"]) > 0

    def test_escalation_package_contains_required_fields(self):
        rag = [{"text": "some evidence", "score": 0.8, "source_collection": "kb_articles",
                "metadata": {"kb_id": "KB-4231", "source_type": "kb_article", "source_file": "KB-4231.md"}}]
        out = escalation_node(self._base_state(grounding_score=0.05, grounding_pass=False, rag_results=rag))
        pkg = out["escalation_package"]
        for field in ("query", "entities", "grounding_score", "escalation_reason"):
            assert field in pkg

    def test_resolved_sets_final_response_to_draft(self):
        out = escalation_node(self._base_state(draft_response="The fix is X."))
        assert out["final_response"] == "The fix is X."

    def test_resolved_with_empty_draft_uses_fallback(self):
        out = escalation_node(self._base_state(draft_response=""))
        assert out["outcome"] == "resolved"
        assert len(out["final_response"]) > 0

    def test_best_effort_resolved_on_partial_grounding(self):
        out = escalation_node(self._base_state(grounding_score=0.5, grounding_pass=False))
        assert out["outcome"] == "resolved"

    def test_clarify_response_format(self):
        # Directly test the clarify branch structure: when outcome is "clarify",
        # the node must include clarification_question in the output.
        # Build a state that matches the clarify branch by patching internal state.
        # Use a safety-free query with no data so we verify the clarify key structure.
        # (Full clarify path requires LLM — see LLM group; this tests the response shape.)
        from src.agents.escalation import escalation_node as _esc
        import unittest.mock as mock
        with mock.patch("src.agents.escalation._get_llm") as mock_llm:
            mock_llm.return_value.invoke.return_value.content = '{"outcome": "clarify", "reason": "too vague"}'
            rag = [{"text": "t", "score": 0.8, "source_collection": "kb_articles", "metadata": {}}]
            out = _esc({
                "query": "something vague",
                "grounding_score": 0.5,
                "evidence_sufficient": False,
                "entities": {},
                "draft_response": "I don't know.",
                "tool_results": [],
                "rag_results": rag,
            })
        assert out["outcome"] == "clarify"
        assert "clarification_question" in out
        assert any(term in out["clarification_question"] for term in ("version", "OS", "GPU"))

    def test_escalation_package_is_none_for_resolved(self):
        out = escalation_node(self._base_state())
        assert out["escalation_package"] is None

    def test_escalation_package_is_none_for_clarify(self):
        import unittest.mock as mock
        with mock.patch("src.agents.escalation._get_llm") as mock_llm:
            mock_llm.return_value.invoke.return_value.content = '{"outcome": "clarify", "reason": "too vague"}'
            rag = [{"text": "t", "score": 0.8, "source_collection": "kb_articles", "metadata": {}}]
            out = escalation_node({
                "query": "something",
                "grounding_score": 0.5,
                "evidence_sufficient": False,
                "entities": {},
                "draft_response": "unknown",
                "tool_results": [],
                "rag_results": rag,
            })
        assert out["outcome"] == "clarify"
        assert out["escalation_package"] is None
