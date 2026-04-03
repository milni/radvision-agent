"""Tests for Phase 5: all five agent nodes.

All tests are pure unit tests — no ChromaDB, no MLflow run required.
MLflow calls inside nodes silently no-op when no run is active.
"""

import pytest

from src.agents.escalation import escalation_node
from src.agents.grounding import grounding_node, _score_claims, _collect_evidence_text
from src.agents.sufficiency import sufficiency_node
from src.agents.synthesizer import synthesizer_node, _kb_refs
from src.agents.triage import (
    triage_node,
    _classify_intent,
    _decide_route,
    _extract_entities,
)


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
# Triage — entity extraction
# ---------------------------------------------------------------------------


class TestTriageEntities:
    def test_extracts_version(self):
        assert _extract_entities("Running v4.2.1 on RHEL")["version"] == "4.2.1"

    def test_extracts_os(self):
        assert _extract_entities("RHEL 9 system")["os"] == "RHEL"

    def test_extracts_gpu(self):
        assert _extract_entities("NVIDIA Quadro RTX card")["gpu"] == "NVIDIA Quadro"

    def test_extracts_component_dicom(self):
        assert _extract_entities("DICOM gateway error")["component"] == "DICOM Gateway"

    def test_extracts_component_fhir(self):
        assert _extract_entities("FHIR endpoint failing")["component"] == "Integration Gateway"

    def test_extracts_component_collab(self):
        assert _extract_entities("screen sharing lag in collab")["component"] == "CollabHub"

    def test_no_entities_empty_query(self):
        assert _extract_entities("some generic question") == {}

    def test_extracts_error_code_tkt(self):
        assert _extract_entities("error TKT-1234 occurred")["error_code"] == "TKT-1234"

    def test_extracts_error_code_kb(self):
        assert _extract_entities("see KB-4231 for reference")["error_code"] == "KB-4231"


class TestTriageIntent:
    def test_troubleshooting(self):
        assert _classify_intent("DICOM connection timeout error") == "troubleshooting"

    def test_feature_inquiry(self):
        assert _classify_intent("what's new in version 4.2") == "feature_inquiry"

    def test_config_help(self):
        assert _classify_intent("how to configure the DICOM gateway setting") == "config_help"

    def test_general_fallback(self):
        assert _classify_intent("tell me about RadVision Pro") == "general"

    def test_troubleshooting_wins_over_feature_inquiry(self):
        # "error" matches troubleshooting; "version" matches feature — troubleshooting takes priority
        assert _classify_intent("error in version 4.2") == "troubleshooting"


class TestTriageRoute:
    def test_error_pattern_for_tls_error(self):
        assert _decide_route("TLS renegotiation error on DICOM", "troubleshooting") == "error_pattern"

    def test_config_check_for_compatibility(self):
        assert _decide_route("Is GPU supported in version 4.2", "general") == "config_check"

    def test_docs_kb_default(self):
        assert _decide_route("cardiac rendering issue", "troubleshooting") == "docs_kb"

    def test_config_check_takes_priority_over_error_pattern(self):
        # config_check is checked first in _decide_route, before error_pattern
        q = "TLS renegotiation error — is GPU supported in v4.2"
        assert _decide_route(q, "troubleshooting") == "config_check"


class TestTriageNode:
    def test_returns_required_keys(self):
        out = triage_node({"query": "DICOM error", "persona": "support"})
        for key in ("entities", "intent", "route_decision", "tool_results", "rag_results"):
            assert key in out

    def test_initialises_empty_results(self):
        out = triage_node({"query": "anything", "persona": "support"})
        assert out["tool_results"] == []
        assert out["rag_results"] == []

    def test_no_crash_on_missing_query(self):
        out = triage_node({"persona": "support"})
        assert out["intent"] == "general"
        assert out["route_decision"] == "docs_kb"

    def test_no_crash_on_missing_persona(self):
        out = triage_node({"query": "DICOM error"})
        assert "intent" in out  # node ran without error


# ---------------------------------------------------------------------------
# Sufficiency Check
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
        from src.config import SUFFICIENCY_MAX_RETRIES
        out = sufficiency_node(self._base_state(retry_count=SUFFICIENCY_MAX_RETRIES))
        assert out["evidence_sufficient"] is False
        assert "max retries" in out["retry_reason"]

    def test_no_match_tool_result_is_insufficient(self):
        out = sufficiency_node(self._base_state(tool_results=[_tool_no_match()]))
        assert out["evidence_sufficient"] is False

    def test_sufficient_with_compat_limited(self):
        # "limited" support status still counts as sufficient evidence
        out = sufficiency_node(self._base_state(tool_results=[_compat_result("limited")]))
        assert out["evidence_sufficient"] is True

    def test_sufficient_sets_empty_retry_reason(self):
        out = sufficiency_node(self._base_state(tool_results=[_tool_match()]))
        assert out["retry_reason"] == ""

    def test_route_fallback_from_error_pattern_suggests_docs_kb(self):
        # insufficient + current route=error_pattern → fallback to docs_kb
        out = sufficiency_node(self._base_state(route="error_pattern"))
        assert "docs_kb" in out["retry_reason"]

    def test_retry_count_incremented_when_retrying(self):
        # Counter must advance so the graph loop cannot run forever
        out = sufficiency_node(self._base_state(retry_count=0))
        assert out["retry_count"] == 1

    def test_retry_count_not_incremented_after_max_retries(self):
        from src.config import SUFFICIENCY_MAX_RETRIES
        out = sufficiency_node(self._base_state(retry_count=SUFFICIENCY_MAX_RETRIES))
        assert out["retry_count"] == SUFFICIENCY_MAX_RETRIES


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------


class TestSynthesizerNode:
    def _base_state(self, persona="support", rag_results=None, tool_results=None):
        return {
            "persona":      persona,
            "tool_results": tool_results or [],
            "rag_results":  rag_results or [_rag_result(text="TLS renegotiation must be disabled on the SCP listener.")],
            "entities":     {"version": "4.2"},
        }

    def test_returns_draft_response(self):
        out = synthesizer_node(self._base_state())
        assert "draft_response" in out
        assert len(out["draft_response"]) > 20

    def test_support_persona_has_steps(self):
        out = synthesizer_node(self._base_state(persona="support"))
        assert "Resolution steps" in out["draft_response"] or "1." in out["draft_response"]

    def test_field_engineer_persona_has_technical_details(self):
        out = synthesizer_node(self._base_state(persona="field_engineer"))
        assert "Technical details" in out["draft_response"]

    def test_sales_persona_has_upgrade_framing(self):
        out = synthesizer_node(self._base_state(persona="sales"))
        assert "upgrade" in out["draft_response"].lower() or "support team" in out["draft_response"].lower()

    def test_tool_match_takes_priority_over_rag(self):
        state = self._base_state(tool_results=[_tool_match()])
        out = synthesizer_node(state)
        assert "TLS renegotiation" in out["draft_response"]

    def test_kb_ref_included_when_available(self):
        out = synthesizer_node(self._base_state(tool_results=[_tool_match(kb_ref="KB-4231")]))
        assert "KB-4231" in out["draft_response"]

    def test_no_evidence_produces_clean_fallback(self):
        out = synthesizer_node({"persona": "support", "tool_results": [], "rag_results": [], "entities": {}})
        assert "No specific evidence found" not in out["draft_response"]
        assert len(out["draft_response"]) > 10

    def test_ticket_id_included_in_refs(self):
        rag = [_rag_result()]
        rag[0]["metadata"] = {"ticket_id": "TKT-5001", "source_type": "past_ticket", "source_file": "TKT-5001.md"}
        out = synthesizer_node({"persona": "support", "tool_results": [], "rag_results": rag, "entities": {}})
        assert "TKT-5001" in out["draft_response"]

    def test_unknown_persona_defaults_to_support_template(self):
        out = synthesizer_node({
            "persona": "unknown",
            "tool_results": [],
            "rag_results": [_rag_result(text="TLS renegotiation fix documented here.")],
            "entities": {},
        })
        assert "Resolution steps" in out["draft_response"] or "1." in out["draft_response"]

    def test_kb_refs_deduplication(self):
        # Same KB-4231 from both tool_result and rag_result — should appear only once
        refs = _kb_refs(
            tool_results=[{"matched": True, "kb_ref": "KB-4231", "description": "TLS"}],
            rag_results=[_rag_result(kb_id="KB-4231")],
        )
        assert refs.count("KB-4231") == 1

    def test_no_evidence_fallback_differs_by_persona(self):
        support_out = synthesizer_node({"persona": "support",        "tool_results": [], "rag_results": [], "entities": {}})
        sales_out   = synthesizer_node({"persona": "sales",          "tool_results": [], "rag_results": [], "entities": {}})
        fe_out      = synthesizer_node({"persona": "field_engineer", "tool_results": [], "rag_results": [], "entities": {}})
        assert support_out["draft_response"] != sales_out["draft_response"]
        assert support_out["draft_response"] != fe_out["draft_response"]


# ---------------------------------------------------------------------------
# Grounding Checker
# ---------------------------------------------------------------------------


class TestScoreClaims:
    def test_kb_ref_in_evidence_is_grounded(self):
        score, ungrounded = _score_claims("See KB-4231 for details.", "kb-4231 is documented here")
        assert score == 1.0
        assert ungrounded == []

    def test_kb_ref_not_in_evidence_is_ungrounded(self):
        score, ungrounded = _score_claims("See KB-9999 for details.", "kb-4231 is here")
        assert score == 0.0
        assert len(ungrounded) == 1

    def test_no_technical_claims_scores_one(self):
        score, _ = _score_claims("Please contact support for assistance.", "some evidence")
        assert score == 1.0

    def test_partial_grounding(self):
        draft = "KB-4231 is the fix. KB-9999 is also relevant."
        evidence = "kb-4231 is documented"
        score, ungrounded = _score_claims(draft, evidence)
        assert 0.0 < score < 1.0
        assert any("KB-9999" in u for u in ungrounded)

    def test_version_token_grounded(self):
        score, ungrounded = _score_claims("Issue affects v4.2 systems.", "v4.2 is documented here")
        assert score == 1.0
        assert ungrounded == []


class TestGroundingNode:
    def test_high_score_passes(self):
        rag = [_rag_result(text="KB-4231 TLS renegotiation fix is documented here.")]
        out = grounding_node({
            "draft_response": "See KB-4231 for the TLS fix.",
            "rag_results": rag, "tool_results": [],
        })
        assert out["grounding_pass"] is True
        assert out["grounding_score"] == 1.0

    def test_ungrounded_claim_flagged(self):
        out = grounding_node({
            "draft_response": "See KB-9999 for details.",
            "rag_results": [_rag_result(text="KB-4231 is the relevant article.")],
            "tool_results": [],
        })
        assert out["grounding_pass"] is False
        assert len(out["ungrounded_claims"]) > 0

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
        assert out["grounding_score"] == 0.0

    def test_regen_count_incremented_on_grounding_fail(self):
        # Counter must advance so the synthesizer → grounding loop cannot run forever
        out = grounding_node({
            "draft_response": "KB-9999 is the fix.",
            "rag_results": [_rag_result(text="KB-4231 is relevant.")],
            "tool_results": [],
            "grounding_regen_count": 0,
        })
        assert out["grounding_pass"] is False
        assert out["grounding_regen_count"] == 1

    def test_regen_count_not_incremented_after_max_regenerations(self):
        from src.config import GROUNDING_MAX_REGENERATIONS
        out = grounding_node({
            "draft_response": "KB-9999 is the fix.",
            "rag_results": [_rag_result(text="KB-4231 is relevant.")],
            "tool_results": [],
            "grounding_regen_count": GROUNDING_MAX_REGENERATIONS,
        })
        assert out["grounding_regen_count"] == GROUNDING_MAX_REGENERATIONS


# ---------------------------------------------------------------------------
# Collect evidence text
# ---------------------------------------------------------------------------


class TestCollectEvidenceText:
    def test_tool_result_values_flattened_and_lowercased(self):
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
# Escalation Gate
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

    def test_escalate_on_critically_low_grounding(self):
        out = escalation_node(self._base_state(grounding_score=0.1, grounding_pass=False))
        assert out["outcome"] == "escalate"

    def test_clarify_when_no_evidence_and_no_entities(self):
        out = escalation_node(self._base_state(
            evidence_sufficient=False, entities={}, grounding_pass=False, grounding_score=0.5,
        ))
        assert out["outcome"] == "clarify"
        assert "clarification_question" in out

    def test_escalate_when_no_evidence_but_entities_present(self):
        """Out-of-scope query: entities extracted but nothing found in KB — must escalate."""
        out = escalation_node(self._base_state(
            evidence_sufficient=False,
            grounding_pass=True,   # synthesizer produced no-claim text → grounding=1.0
            grounding_score=1.0,
            entities={"version": "4.2"},  # entity present → not "clarify"
        ))
        assert out["outcome"] == "escalate"

    def test_escalation_package_contains_required_fields(self):
        out = escalation_node(self._base_state(grounding_score=0.1, grounding_pass=False))
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
        # evidence found but grounding_pass=False and score above escalation threshold → best-effort resolved
        out = escalation_node(self._base_state(grounding_score=0.5, grounding_pass=False))
        assert out["outcome"] == "resolved"

    def test_clarify_lists_missing_context_in_question(self):
        out = escalation_node(self._base_state(
            evidence_sufficient=False, entities={}, grounding_pass=False, grounding_score=0.5,
        ))
        question = out["clarification_question"]
        assert any(term in question for term in ("Missing", "version", "OS", "GPU"))

    def test_escalation_package_is_none_for_resolved(self):
        out = escalation_node(self._base_state())
        assert out["escalation_package"] is None

    def test_escalation_package_is_none_for_clarify(self):
        out = escalation_node(self._base_state(
            evidence_sufficient=False, entities={}, grounding_pass=False, grounding_score=0.5,
        ))
        assert out["escalation_package"] is None
