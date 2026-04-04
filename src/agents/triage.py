"""Triage + Route node (Node 1) for the RadVision Pro support agent.

Extracts entities from the query, classifies intent, and decides which
source to check first via an LLM call (gemma3:4b by default).

Route decisions:
  error_pattern — specific error signature detected → Log Analyzer first
  config_check  — compatibility/support question   → Compat Checker first
  docs_kb       — everything else                  → RAG subgraph first
"""

import json
import logging

from langchain_ollama import ChatOllama

from src.config import LLM_MODEL, OLLAMA_BASE_URL
from src.tracking import log_params_safe, set_tags_safe

logger = logging.getLogger(__name__)

_TRIAGE_PROMPT = """\
You are the triage node of a radiology software support agent.
Given a support query, return a JSON object with these exact keys:

{{
  "entities": {{
    "version": "<version string or null>",
    "os": "<OS name or null>",
    "gpu": "<GPU model or null>",
    "error_code": "<KB-XXXX / ERR-XXX / TKT-XXXX or null>",
    "component": "<DICOM Gateway | Integration Gateway | Rendering Engine | CollabHub or null>"
  }},
  "intent": "<troubleshooting | feature_inquiry | config_help | general>",
  "route_decision": "<error_pattern | config_check | docs_kb>"
}}

Route rules:
- error_pattern: query contains a SPECIFIC error code, log token, or exact log signature copied
  from a log file. Generic symptoms or HTTP status codes alone do NOT qualify.
  Examples: "AssocReject 0x0006", "VRAM_FALLBACK_TRIGGERED", "HL7 thread_pool_full"
  NOT error_pattern: "FHIR returns 503", "images look weird", "something crashed"
- config_check:  query asks about GPU, OS, or version support/compatibility, OR asks which
  features work/are available on a specific GPU or platform. Applies even when phrased as
  a sales question ("what should I tell them", "can we deploy", "will it work for a client").
  Examples: "Does T4 support Cardiac 4D?", "What features work on AMD MI210?",
  "Is RHEL 9 supported?", "Can we deploy on this GPU?", "Which features will work?",
  "A prospect has T4 GPUs and wants Cardiac 4D", "What should I tell them about GPU support?"
- docs_kb:       everything else — how-to, configuration steps, troubleshooting without a
  specific log token, release notes, product overview, feature descriptions

Query: {query}

Return only the JSON object, no explanation."""

_llm = None


def _get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0, format="json")
    return _llm


def triage_node(state: dict) -> dict:
    """Extract entities, classify intent, and pick a route via LLM."""
    query: str = state.get("query", "")
    persona: str = state.get("persona", "support")

    response = _get_llm().invoke(_TRIAGE_PROMPT.format(query=query))
    data = json.loads(response.content)

    entities = {k: v for k, v in data.get("entities", {}).items() if v}
    intent = data.get("intent", "general")
    route_decision = data.get("route_decision", "docs_kb")
    if route_decision not in ("error_pattern", "config_check", "docs_kb"):
        route_decision = "docs_kb"
    logger.info("Triage: intent=%s route=%s entities=%s", intent, route_decision, entities)
    log_params_safe({"persona": persona})
    set_tags_safe({"intent": intent, "route_decision": route_decision})

    return {
        "entities":       entities,
        "intent":         intent,
        "route_decision": route_decision,
        "tool_results":   [],
        "rag_results":    [],
        "retry_count":    state.get("retry_count", 0),
    }
