"""Query Rewriter node for the RAG subgraph.

Expands the query with domain synonyms to improve retrieval recall.
Rule-based — no LLM call needed — so the subgraph works without Ollama.

On a retry (rewrite_count > 0) a generic troubleshooting fallback is also
appended so the second attempt casts a wider net.
"""

import logging
import re

logger = logging.getLogger(__name__)

# (regex_pattern, space-separated expansion terms to append)
_EXPANSIONS: list[tuple[str, str]] = [
    (r"\bDICOM\b",                      "DIMSE SCP SCU association"),
    (r"\bTLS\b|\bSSL\b",               "SSL certificate renegotiation handshake"),
    (r"\btimeout\b",                    "connection timeout association timeout"),
    (r"\bcrash\b|\bfail\b",            "failure error exception crash"),
    (r"\bslow\b|\blag\b|\blatency\b",  "performance degraded slow latency"),
    (r"\bGPU\b|\bVRAM\b",              "VRAM graphics card memory rendering"),
    (r"\bFHIR\b",                       "HL7 integration endpoint REST"),
    (r"\brendering\b|\bartifact\b",     "3D 4D visualization rendering artifacts"),
    (r"\bscreen.shar\b|\bcollabor\b|\bWebRTC\b", "WebRTC collaboration sharing latency"),
    (r"\b503\b|service unavailable",   "connection pool overload 503"),
    (r"\bconfigur\b|\bsett\b|\bparam\b", "configuration parameter value setting"),
]


def rewrite_query(state: dict) -> dict:
    """Expand the query with domain synonyms to improve retrieval recall.

    Reads state['query'] (the original, never modified) and produces
    state['rewritten_query'] by appending relevant synonym terms.
    Existing words are not duplicated.
    """
    query: str = state["query"]
    rewrite_count: int = state.get("rewrite_count", 0)

    additions: list[str] = []
    for pattern, expansion in _EXPANSIONS:
        if re.search(pattern, query, re.IGNORECASE):
            additions.append(expansion)

    # On retry: append a broad fallback to widen the search
    if rewrite_count > 0:
        additions.append("error issue problem troubleshooting support")

    existing_lower = {w.lower() for w in query.split()}
    new_terms = [t for t in " ".join(additions).split() if t.lower() not in existing_lower]

    rewritten = (query + " " + " ".join(new_terms)).strip() if new_terms else query
    logger.debug("Rewriter (attempt %d): %r -> %r", rewrite_count + 1, query, rewritten)

    return {"rewritten_query": rewritten}
