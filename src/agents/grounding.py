"""Grounding Checker node (Node 4) for the RadVision Pro support agent.

Verifies that technical claims in the draft response are backed by the
gathered evidence (rag_results + tool_results).

Scoring (dummy LLM):
  1. Extract "claim sentences" — sentences that contain at least one
     technical token: KB-XXXX, TKT-XXXX, version numbers, known setting
     names, or component names.
  2. A claim is grounded if any of its technical tokens appear in the
     combined evidence text.
  3. grounding_score = grounded_claims / total_claims
     (if no claims detected, score = 1.0 — nothing to contradict)
  4. grounding_pass = score >= GROUNDING_THRESHOLD

Real LLM swap: replace _score_claims() with a structured prompt that
cross-references each sentence against the evidence chunks.
"""

import logging
import re

from src.config import GROUNDING_THRESHOLD
from src.tracking import log_metrics_safe, set_tags_safe

logger = logging.getLogger(__name__)

# Tokens that mark a sentence as a verifiable technical claim
_CLAIM_TOKEN_RE = re.compile(
    r"\b(KB-\d+|TKT-\d{4,}|v?\d+\.\d+(?:\.\d+)?"
    r"|SCP_\w+|DICOM|FHIR|VRAM|TLS|WebRTC|Cardiac 4D|CollabHub"
    r"|DICOM Gateway|Rendering Engine|Integration Gateway)\b",
    re.IGNORECASE,
)


def _collect_evidence_text(tool_results: list[dict], rag_results: list[dict]) -> str:
    """Flatten all tool and RAG evidence into a single lowercase string.

    Includes both text content and metadata values so that KB IDs, version
    numbers, and component names in metadata are also searchable.
    Returns an empty string when both result lists are empty.
    """
    parts: list[str] = []
    for r in tool_results:
        parts.extend([str(v) for v in r.values() if v])
    for r in rag_results:
        parts.append(r.get("text", ""))
        parts.extend(str(v) for v in r.get("metadata", {}).values() if v)
    return " ".join(parts).lower()


def _score_claims(
    draft: str, evidence: str
) -> tuple[float, list[str]]:
    """Return (score, ungrounded_claim_sentences)."""
    sentences = [s.strip() for s in re.split(r"[.!?\n]", draft) if s.strip()]

    claim_sentences = [s for s in sentences if _CLAIM_TOKEN_RE.search(s)]
    if not claim_sentences:
        return 1.0, []

    ungrounded: list[str] = []
    for sentence in claim_sentences:
        tokens = _CLAIM_TOKEN_RE.findall(sentence)
        grounded = any(token.lower() in evidence for token in tokens)
        if not grounded:
            ungrounded.append(sentence)

    score = (len(claim_sentences) - len(ungrounded)) / len(claim_sentences)
    return round(score, 3), ungrounded


def grounding_node(state: dict) -> dict:
    """Score the draft response against evidence; flag ungrounded claims."""
    draft:             str        = state.get("draft_response", "")
    tool_results:      list[dict] = state.get("tool_results", [])
    rag_results:       list[dict] = state.get("rag_results", [])
    regen_count:       int        = state.get("grounding_regen_count", 0)

    # An empty draft has nothing to verify but also nothing useful — treat as failing.
    if not draft.strip():
        logger.warning("Grounding: empty draft_response — failing grounding")
        return {
            "grounding_score":       0.0,
            "ungrounded_claims":     [],
            "grounding_pass":        False,
            "final_response":        "",
            "grounding_regen_count": regen_count,  # passed through; graph node increments
        }

    evidence = _collect_evidence_text(tool_results, rag_results)
    score, ungrounded = _score_claims(draft, evidence)
    grounding_pass = score >= GROUNDING_THRESHOLD

    logger.info(
        "Grounding: score=%.3f pass=%s ungrounded=%d regen_count=%d",
        score, grounding_pass, len(ungrounded), regen_count,
    )

    log_metrics_safe({
        "grounding.score":            score,
        "grounding.ungrounded_count": len(ungrounded),
    })
    set_tags_safe({"grounding_pass": str(grounding_pass)})

    return {
        "grounding_score":       score,
        "ungrounded_claims":     ungrounded,
        "grounding_pass":        grounding_pass,
        "grounding_regen_count": regen_count,  # passed through; graph node increments on fail
        "final_response":        draft,         # promoted to final; may be replaced on regen
    }
