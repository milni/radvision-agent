"""Triage + Route node (Node 1) for the RadVision Pro support agent.

Extracts entities from the query, classifies intent, and decides which
source to check first.

Route decisions:
  error_pattern — specific error signature detected → Log Analyzer first
  config_check  — compatibility/support question   → Compat Checker first
  docs_kb       — everything else                  → RAG subgraph first

Dummy LLM: fully rule-based (regex + keyword matching).
Real LLM swap: replace _dummy_triage() with a structured JSON prompt call.
"""

import logging
import re

from src.tracking import log_params_safe, set_tags_safe

logger = logging.getLogger(__name__)

# --- Entity extraction ---

_VERSION_RE = re.compile(r"\bv?(\d+\.\d+(?:\.\d+)?)\b")
_ERROR_CODE_RE = re.compile(r"\b(TKT-\d{4,}|KB-\d{4,}|ERR-\d+)\b", re.IGNORECASE)

_OS_KEYWORDS = {
    "windows": "Windows", "win ": "Windows",
    "rhel": "RHEL", "red hat": "RHEL",
    "ubuntu": "Ubuntu", "centos": "CentOS", "debian": "Debian",
    "macos": "macOS", "mac os": "macOS",
}
_GPU_KEYWORDS = {
    "quadro": "NVIDIA Quadro", "rtx": "NVIDIA RTX",
    "nvidia": "NVIDIA", "amd radeon": "AMD Radeon",
    "amd": "AMD", "intel": "Intel",
}
_COMPONENT_KEYWORDS = {
    "dicom": "DICOM Gateway", "fhir": "Integration Gateway",
    "rendering": "Rendering Engine", "4d rendering": "Rendering Engine",
    "collab": "CollabHub", "screen shar": "CollabHub", "webrtc": "CollabHub",
}

# --- Intent classification ---

_TROUBLESHOOTING_RE = re.compile(
    r"\berror\b|\bfail\b|\bcrash\b|\btimeout\b|\breject\b|\bartifact\b"
    r"|\blag\b|\bslow\b|\b503\b|\bnot work\b|\bissue\b|\bproblem\b",
    re.IGNORECASE,
)
_FEATURE_RE = re.compile(
    r"\bversion\b|\brelease\b|\bwhat.?s new\b|\bfeature\b|\bchangelog\b|\bnew in\b",
    re.IGNORECASE,
)
_CONFIG_RE = re.compile(
    r"\bconfigur\b|\bsetting\b|\bparam\b|\bhow.?to\b|\bsetup\b|\binstall\b|\benable\b",
    re.IGNORECASE,
)

# --- Route selection ---

# Specific error signatures → Log Analyzer has a pattern for these
_ERROR_PATTERN_RE = re.compile(
    r"\bTLS renegotiation\b|\bDICOM association\b|\bSCP connection\b"
    r"|\brendering artifact\b|\bVRAM\b|\bconnection pool\b|\bscreen sharing lag\b",
    re.IGNORECASE,
)
# Compatibility/support questions → Compat Checker
_CONFIG_CHECK_RE = re.compile(
    r"\bcompatible\b|\bsupported?\b.{0,25}\b(GPU|OS|version|platform)\b"
    r"|\b(GPU|OS|version|platform)\b.{0,25}\bsupported?\b"
    r"|\bworks? with\b",
    re.IGNORECASE,
)


def _extract_entities(query: str) -> dict:
    """Extract structured entities from the query using regex and keyword matching.

    Returns a dict with any subset of: version, os, gpu, error_code, component.
    Missing fields are simply absent — callers must use .get() with a default.
    """
    q_lower = query.lower()
    entities: dict = {}

    m = _VERSION_RE.search(query)
    if m:
        entities["version"] = m.group(1)

    for kw, label in _OS_KEYWORDS.items():
        if kw in q_lower:
            entities["os"] = label
            break

    for kw, label in _GPU_KEYWORDS.items():
        if kw in q_lower:
            entities["gpu"] = label
            break

    m = _ERROR_CODE_RE.search(query)
    if m:
        entities["error_code"] = m.group(1).upper()

    for kw, label in _COMPONENT_KEYWORDS.items():
        if kw in q_lower:
            entities["component"] = label
            break

    return entities


def _classify_intent(query: str) -> str:
    """Classify the query into one of four intents.

    Priority order: troubleshooting > feature_inquiry > config_help > general.
    The first matching pattern wins; falls back to "general" if none match.
    """
    if _TROUBLESHOOTING_RE.search(query):
        return "troubleshooting"
    if _FEATURE_RE.search(query):
        return "feature_inquiry"
    if _CONFIG_RE.search(query):
        return "config_help"
    return "general"


def _decide_route(query: str, intent: str) -> str:
    """Choose which tool or subgraph to invoke first.

    config_check takes priority (compatibility questions are unambiguous).
    error_pattern is chosen only when intent is troubleshooting AND a known
    error signature is present. Everything else goes to docs_kb (RAG subgraph).
    """
    if _CONFIG_CHECK_RE.search(query):
        return "config_check"
    if intent == "troubleshooting" and _ERROR_PATTERN_RE.search(query):
        return "error_pattern"
    return "docs_kb"


def triage_node(state: dict) -> dict:
    """Extract entities, classify intent, and pick a route."""
    query: str = state.get("query", "")
    persona: str = state.get("persona", "support")

    entities = _extract_entities(query)
    intent = _classify_intent(query)
    route_decision = _decide_route(query, intent)

    logger.info("Triage: intent=%s route=%s entities=%s", intent, route_decision, entities)

    log_params_safe({"persona": persona})
    set_tags_safe({"intent": intent, "route_decision": route_decision})

    return {
        "entities": entities,
        "intent": intent,
        "route_decision": route_decision,
        "tool_results": [],
        "rag_results": [],
        "retry_count": state.get("retry_count", 0),
    }
