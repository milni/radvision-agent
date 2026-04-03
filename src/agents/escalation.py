"""Escalation Gate node (Node 5) for the RadVision Pro support agent.

Three-way decision based on grounding quality and evidence completeness:

  resolved  — grounding passed AND evidence was sufficient
  clarify   — evidence insufficient AND the query lacked key context
              (no version, no OS, no component identified by triage)
  escalate  — safety/compliance keywords detected, OR grounding score
              critically low (< 0.3), OR all other conditions fail

Packages an escalation_package dict when outcome == "escalate" so a
human expert has everything needed to continue the case.
"""

import logging
import re

from src.config import GROUNDING_THRESHOLD
from src.tracking import set_tags_safe

logger = logging.getLogger(__name__)

_SAFETY_RE = re.compile(
    r"\bsafety\b|\bpatient\b|\bcritical.{0,10}(care|system)\b"
    r"|\bemergency\b|\bHIPAA\b|\bcompliance\b|\bregulat\b",
    re.IGNORECASE,
)

_ESCALATION_THRESHOLD = 0.3   # below this → always escalate


def escalation_node(state: dict) -> dict:
    """Decide the final outcome and build the response or escalation package."""
    query:              str   = state.get("query", "")
    grounding_score:    float = state.get("grounding_score", 0.0)
    grounding_pass:     bool  = state.get("grounding_pass", False)
    evidence_sufficient: bool = state.get("evidence_sufficient", False)
    entities:           dict  = state.get("entities", {})
    draft:              str   = state.get("draft_response", "")

    # --- Outcome logic ---
    if _SAFETY_RE.search(query):
        outcome = "escalate"
        reason  = "safety or compliance keywords detected in query"

    elif grounding_score < _ESCALATION_THRESHOLD:
        outcome = "escalate"
        reason  = f"grounding score critically low ({grounding_score:.2f} < {_ESCALATION_THRESHOLD})"

    elif grounding_pass and evidence_sufficient:
        outcome = "resolved"
        reason  = "sufficient grounded evidence"

    elif not evidence_sufficient and not entities:
        outcome = "clarify"
        reason  = "query lacks context (no version, OS, GPU, or component identified)"

    elif not evidence_sufficient:
        # Entities were extracted but nothing relevant was found after all retries.
        # Don't pretend we have an answer — escalate to a specialist.
        outcome = "escalate"
        reason  = "no relevant evidence found in knowledge base after retries"

    else:
        # Evidence found but grounding is partial — best-effort resolve is acceptable.
        outcome = "resolved"
        reason  = "partial evidence; best-effort response"

    logger.info("Escalation: outcome=%s — %s", outcome, reason)
    set_tags_safe({"outcome": outcome})

    # --- Build output ---
    if outcome == "resolved":
        final = draft if draft.strip() else (
            "The agent processed your query but could not produce a response. "
            "Please contact support directly."
        )
        return {
            "outcome":            outcome,
            "final_response":     final,
            "escalation_package": None,
        }

    if outcome == "clarify":
        missing = [k for k in ("version", "os", "gpu", "component") if k not in entities]
        question = (
            "Could you provide more details? "
            + (f"Missing: {', '.join(missing)}. " if missing else "")
            + "For example: which version of RadVision Pro, OS, and GPU are you using?"
        )
        return {
            "outcome":               outcome,
            "clarification_question": question,
            "final_response":        question,
            "escalation_package":    None,
        }

    # escalate
    package = {
        "query":           query,
        "entities":        entities,
        "tool_results":    state.get("tool_results", []),
        "rag_results":     [
            {"score": r["score"], "source": r["source_collection"],
             "kb_id": r["metadata"].get("kb_id", ""), "text": r["text"][:300]}
            for r in state.get("rag_results", [])
        ],
        "grounding_score": grounding_score,
        "draft_response":  draft,
        "escalation_reason": reason,
    }
    return {
        "outcome":            outcome,
        "final_response":     (
            "This case has been escalated to a specialist. "
            f"Reason: {reason}. You will be contacted shortly."
        ),
        "escalation_package": package,
    }
