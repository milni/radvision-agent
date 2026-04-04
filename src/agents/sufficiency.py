"""Sufficiency Check node (Node 2) for the RadVision Pro support agent.

Evaluates whether the gathered evidence (tool_results + rag_results) is
enough to answer the query. If not and retries remain, suggests an
alternative route for the triage node to try next.

Heuristics (dummy LLM):
  sufficient = any log-analyzer match
              OR any compat-checker supported/limited result
              OR any RAG result at or above RAG_RELEVANCE_THRESHOLD
"""

import logging

from src.config import RAG_RELEVANCE_THRESHOLD, SUFFICIENCY_MAX_RETRIES
from src.tracking import log_metrics_safe, set_tags_safe

logger = logging.getLogger(__name__)

# When evidence is insufficient, suggest the next route to try
_ROUTE_FALLBACK = {
    "error_pattern": "docs_kb",
    "config_check":  "docs_kb",
    "docs_kb":       "error_pattern",
}


def sufficiency_node(state: dict) -> dict:
    """Decide whether current evidence is sufficient to generate a response."""
    tool_results: list[dict] = state.get("tool_results", [])
    rag_results:  list[dict] = state.get("rag_results", [])
    retry_count:  int        = state.get("retry_count", 0)

    has_log_match   = any(r.get("matched") for r in tool_results)
    has_compat      = any(r.get("status") in ("supported", "limited", "unsupported", "experimental")
                          for r in tool_results)
    top_rag_score   = rag_results[0]["score"] if rag_results else 0.0
    has_rag         = top_rag_score >= RAG_RELEVANCE_THRESHOLD

    sufficient   = has_log_match or has_compat or has_rag
    retry_reason = ""

    if not sufficient:
        if retry_count < SUFFICIENCY_MAX_RETRIES:
            current_route = state.get("route_decision", "docs_kb")
            fallback      = _ROUTE_FALLBACK.get(current_route, "docs_kb")
            retry_reason  = (
                f"No matching evidence via '{current_route}'. "
                f"Retrying with '{fallback}'."
            )
            retry_count += 1   # consumed one retry — graph must not loop forever
            logger.info("Sufficiency: INSUFFICIENT — %s", retry_reason)
        else:
            retry_reason = "Evidence insufficient after max retries. Proceeding."
            logger.info("Sufficiency: %s", retry_reason)
    else:
        logger.info(
            "Sufficiency: SUFFICIENT (log=%s compat=%s rag=%.3f)",
            has_log_match, has_compat, top_rag_score,
        )

    log_metrics_safe({
        "sufficiency.top_rag_score": round(top_rag_score, 4),
        "sufficiency.retry_count":   retry_count,
    })
    set_tags_safe({"evidence_sufficient": str(sufficient)})

    return {
        "evidence_sufficient": sufficient,
        "retry_reason":        retry_reason,
        "retry_count":         retry_count,
    }
