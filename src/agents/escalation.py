"""Escalation Gate node (Node 5) for the RadVision Pro support agent.

Three-way decision via LLM (gemma3:1b):
  resolved  — the draft answers the query with sufficient grounded evidence
  clarify   — query is too vague (missing version/OS/component)
  escalate  — safety/compliance issue, critical error, or no useful evidence

Safety regex and critically-low grounding score bypass the LLM — those
are hard rules that cannot be overridden.

Packages an escalation_package dict when outcome == "escalate".
"""

import json
import logging
import re

from langchain_ollama import ChatOllama

from src.config import ESCALATION_LLM_MODEL, OLLAMA_BASE_URL
from src.tracking import set_tags_safe

logger = logging.getLogger(__name__)

_SAFETY_RE = re.compile(
    r"\bsafety\b|\bpatient\b|\bcritical.{0,10}(care|system)\b"
    r"|\bemergency\b|\bHIPAA\b|\bcompliance\b|\bregulat\b",
    re.IGNORECASE,
)

_ESCALATION_THRESHOLD = 0.3

_ESCALATION_PROMPT = """\
You are the escalation gate of a radiology software support agent.
Decide the outcome for this support interaction.

Query: {query}
Entities extracted: {entities}
Evidence sufficient: {evidence_sufficient}
Grounding score: {grounding_score}
Draft response: {draft}

Choose exactly one outcome:
- "resolved"  — the draft answers the query with sufficient grounded evidence
- "clarify"   — the query is too vague to answer (missing version/OS/component info)
- "escalate"  — safety/compliance issue, critical error with no KB article, or no useful evidence found

Return a JSON object: {{"outcome": "<resolved|clarify|escalate>", "reason": "<one sentence>"}}
Return only the JSON object."""

_llm = None


def _get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(model=ESCALATION_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0, format="json")
    return _llm


def escalation_node(state: dict) -> dict:
    """Decide the final outcome and build the response or escalation package."""
    query:               str   = state.get("query", "")
    grounding_score:     float = state.get("grounding_score", 0.0)
    evidence_sufficient: bool  = state.get("evidence_sufficient", False)
    entities:            dict  = state.get("entities", {})
    draft:               str   = state.get("draft_response", "")

    # Hard rules — no LLM override
    if _SAFETY_RE.search(query):
        outcome = "escalate"
        reason  = "safety or compliance keywords detected in query"

    elif grounding_score < _ESCALATION_THRESHOLD:
        outcome = "escalate"
        reason  = f"grounding score critically low ({grounding_score:.2f} < {_ESCALATION_THRESHOLD})"

    else:
        response = _get_llm().invoke(_ESCALATION_PROMPT.format(
            query=query[:300],
            entities=entities,
            evidence_sufficient=evidence_sufficient,
            grounding_score=round(grounding_score, 2),
            draft=draft[:400],
        ))
        data = json.loads(response.content)
        outcome = data.get("outcome", "resolved")
        if outcome not in ("resolved", "clarify", "escalate"):
            outcome = "resolved"
        reason = data.get("reason", "")

    logger.info("Escalation: outcome=%s — %s", outcome, reason)
    set_tags_safe({"outcome": outcome})

    if outcome == "resolved":
        return {
            "outcome":            outcome,
            "final_response":     draft or "The agent processed your query but could not produce a response. Please contact support directly.",
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
            "outcome":                outcome,
            "clarification_question": question,
            "final_response":         question,
            "escalation_package":     None,
        }

    # escalate
    package = {
        "query":             query,
        "entities":          entities,
        "tool_results":      state.get("tool_results", []),
        "rag_results":       [
            {"score": r["score"], "source": r["source_collection"],
             "kb_id": r["metadata"].get("kb_id", ""), "text": r["text"][:300]}
            for r in state.get("rag_results", [])
        ],
        "grounding_score":   grounding_score,
        "draft_response":    draft,
        "escalation_reason": reason,
    }
    return {
        "outcome":            outcome,
        "final_response":     f"This case has been escalated to a specialist. Reason: {reason}. You will be contacted shortly.",
        "escalation_package": package,
    }
