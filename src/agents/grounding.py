"""Grounding Checker node (Node 4) for the RadVision Pro support agent.

Verifies that technical claims in the draft response are backed by the
gathered evidence (rag_results + tool_results) via an LLM call (gemma3:1b).

Returns:
  grounding_score       — 0.0–1.0 fraction of supported claim sentences
  ungrounded_claims     — sentences not supported by evidence
  grounding_pass        — score >= GROUNDING_THRESHOLD
"""

import json
import logging

from langchain_ollama import ChatOllama

from src.config import GROUNDING_LLM_MODEL, GROUNDING_THRESHOLD, OLLAMA_BASE_URL
from src.tracking import log_metrics_safe, set_tags_safe

logger = logging.getLogger(__name__)

_GROUNDING_PROMPT = """\
You are a grounding checker. Given a drafted response and supporting evidence, \
identify any sentences in the response that make specific technical claims \
(version numbers, config settings, KB references, error codes) \
that are NOT supported by the evidence.

Evidence:
{evidence}

Draft response:
{draft}

Return a JSON object:
{{"score": <0.0-1.0>, "ungrounded": [<list of unsupported sentences>]}}

score = fraction of technical claim sentences that ARE supported. \
If there are no technical claims, score = 1.0 and ungrounded = [].
Return only the JSON object."""

_llm = None


def _get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(model=GROUNDING_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0, format="json")
    return _llm


def _collect_evidence_text(tool_results: list[dict], rag_results: list[dict]) -> str:
    parts: list[str] = []
    for r in tool_results:
        parts.extend([str(v) for v in r.values() if v])
    for r in rag_results:
        parts.append(r.get("text", ""))
        parts.extend(str(v) for v in r.get("metadata", {}).values() if v)
    return " ".join(parts).lower()


def grounding_node(state: dict) -> dict:
    """Score the draft response against evidence; flag ungrounded claims."""
    draft:        str        = state.get("draft_response", "")
    tool_results: list[dict] = state.get("tool_results", [])
    rag_results:  list[dict] = state.get("rag_results", [])
    regen_count:  int        = state.get("grounding_regen_count", 0)

    if not draft.strip():
        logger.warning("Grounding: empty draft_response — failing grounding")
        return {
            "grounding_score":       0.0,
            "ungrounded_claims":     [],
            "grounding_pass":        False,
            "final_response":        "",
            "grounding_regen_count": regen_count,
        }

    evidence = _collect_evidence_text(tool_results, rag_results)
    response = _get_llm().invoke(_GROUNDING_PROMPT.format(
        evidence=evidence[:1000], draft=draft[:800]
    ))
    data = json.loads(response.content)
    score = round(float(data.get("score", 1.0)), 3)
    ungrounded = data.get("ungrounded", [])
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
        "grounding_regen_count": regen_count,
        "final_response":        draft,
    }
