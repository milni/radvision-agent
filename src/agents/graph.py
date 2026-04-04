"""Main agent graph for the RadVision Pro support agent (Phase 6).

Topology
--------
START → triage
triage → [route_decision]
    error_pattern → run_log_analyzer  ─┐
    config_check  → run_compat_checker ├→ sufficiency
    docs_kb       → run_rag           ─┘
sufficiency → [evidence_sufficient / retry_reason]
    sufficient              → synthesizer
    retry available         → run_log_analyzer | run_rag  (fallback route)
    max retries exhausted   → synthesizer
synthesizer → grounding
grounding → [grounding_pass / grounding_regen_count]
    pass                    → escalation
    fail + regen available  → increment_regen → synthesizer
    fail + max regen        → escalation
escalation → END

Infinite-loop guards
--------------------
Sufficiency retry:
  retry_count is incremented inside sufficiency_node when scheduling a retry.
  _route_from_sufficiency routes to the fallback tool only when "Retrying"
  appears in retry_reason; once max retries are reached the text changes to
  "max retries" and the edge falls through to synthesizer.
  Cap: SUFFICIENCY_MAX_RETRIES = 1 (one extra tool call, then forced proceed).

Grounding regen:
  grounding_node passes grounding_regen_count through UNCHANGED.
  The increment_regen node bumps it before routing back to synthesizer.
  _route_from_grounding checks regen_count < GROUNDING_MAX_REGENERATIONS;
  once at the cap the edge routes to escalation instead.
  Cap: GROUNDING_MAX_REGENERATIONS = 1 (one re-synthesis, then forced escalate).
"""

import logging
from pathlib import Path
from typing import Literal

import mlflow
from langgraph.graph import END, START, StateGraph

from src.agents.escalation import escalation_node
from src.agents.grounding import grounding_node
from src.agents.sufficiency import sufficiency_node
from src.agents.synthesizer import synthesizer_node
from src.agents.triage import triage_node
from src.config import (
    GROUNDING_MAX_REGENERATIONS,
    LLM_MODEL,
    RAG_TOP_K,
    VECTORSTORE_DIR,
)
from src.rag.subgraph import build_rag_subgraph
from src.state import AgentState
from src.tools.compat_checker import CompatibilityChecker
from src.tools.log_analyzer import LogAnalyzer
from src.tracking import (
    EXPERIMENT_AGENT,
    log_metrics_safe,
    log_params_safe,
    set_experiment,
    set_tags_safe,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool executor nodes
# ---------------------------------------------------------------------------

# Loaded once at module import — avoids re-parsing YAML on every call.
_log_analyzer = LogAnalyzer()
_compat_checker = CompatibilityChecker()

# Triage component names → specific feature to check when the component is known
_COMPONENT_TO_FEATURE = {
    "Rendering Engine":    "Cardiac 4D",
    "Integration Gateway": "FHIR R4",
    "CollabHub":           "WebRTC",
    "DICOM Gateway":       "DICOM TLS",
}

# All features in the compat matrix — used when no specific feature is mentioned
_ALL_FEATURES = ["MPR", "MIP", "VRT", "Cardiac 4D", "Vessel Analysis"]


def _run_log_analyzer(state: dict) -> dict:
    """Run the log pattern analyzer against the full user query."""
    result = _log_analyzer.analyze_log(state.get("query", "")).model_dump()
    logger.info("LogAnalyzer: matched=%s kb_ref=%s", result.get("matched"), result.get("kb_ref"))
    return {"tool_results": [result], "rag_results": []}


def _run_compat_checker(state: dict) -> dict:
    """Run the compatibility checker using entities extracted by triage.

    When a specific component is known, checks that one feature.
    When only a GPU is known (e.g. "which features work on AMD MI210?"),
    checks every feature in the matrix so the synthesizer gets a full picture.
    Falls back to _DEFAULT_VERSION when no version was extracted.
    """
    entities  = state.get("entities", {})
    version   = entities.get("version") or ""
    gpu       = entities.get("gpu", "")
    os_name   = entities.get("os", "")

    results: list[dict] = []

    if gpu:
        # Determine which versions to check.
        # If the user stated a version, use only that.
        # If no version was mentioned, check every version in the matrix for this GPU —
        # never assume or invent a version.
        if version:
            versions_to_check = [version]
        else:
            versions_to_check = _compat_checker.versions_for_gpu(gpu)

        if versions_to_check:
            for v in versions_to_check:
                for feature in _ALL_FEATURES:
                    r = _compat_checker.check_compatibility(v, gpu, feature).model_dump()
                    r["checked_feature"] = feature
                    r["checked_version"] = v
                    results.append(r)
        else:
            results = [{"status": "unknown",
                        "reason": f"No compatibility data found for GPU '{gpu}'.",
                        "recommendation": "Verify the GPU model name and consult release notes."}]

    if os_name:
        results.append(_compat_checker.check_platform(version, os_name).model_dump())

    if not results:
        results = [{"status": "unknown",
                    "reason": "No GPU or OS found in query — cannot run compatibility check.",
                    "recommendation": "Provide the GPU model or OS name to get a compatibility result."}]

    logger.info("CompatChecker: %d results, statuses=%s",
                len(results), [r.get("status") for r in results])
    return {"tool_results": results, "rag_results": []}


# ---------------------------------------------------------------------------
# Grounding regeneration counter node
# ---------------------------------------------------------------------------


def _increment_grounding_regen(state: dict) -> dict:
    """Bump grounding_regen_count before routing back to synthesizer for regen."""
    count = state.get("grounding_regen_count", 0) + 1
    logger.debug("Grounding regen: count → %d", count)
    return {"grounding_regen_count": count}


# ---------------------------------------------------------------------------
# Conditional edge routers
# ---------------------------------------------------------------------------


def _route_from_triage(
    state: dict,
) -> Literal["run_log_analyzer", "run_rag", "run_compat_checker"]:
    """Map route_decision to the first tool or subgraph to invoke."""
    return {
        "error_pattern": "run_log_analyzer",
        "config_check":  "run_compat_checker",
        "docs_kb":       "run_rag",
    }.get(state.get("route_decision", "docs_kb"), "run_rag")


def _route_from_sufficiency(
    state: dict,
) -> Literal["synthesizer", "run_log_analyzer", "run_rag"]:
    """Route after the sufficiency check.

    Sufficient evidence OR max retries exhausted → synthesizer.
    Active retry scheduled → fallback tool (opposite of first attempt).
    """
    if state.get("evidence_sufficient"):
        return "synthesizer"
    # "Retrying" appears in retry_reason only for an active retry.
    # "max retries" means we proceed to synthesis with whatever was gathered.
    if "Retrying" in state.get("retry_reason", ""):
        fallback = {
            "docs_kb":       "run_log_analyzer",
            "error_pattern": "run_rag",
            "config_check":  "run_rag",
        }
        return fallback.get(state.get("route_decision", "docs_kb"), "synthesizer")
    return "synthesizer"


def _route_from_grounding(
    state: dict,
) -> Literal["escalation", "increment_regen"]:
    """Pass → escalation.  Fail + regen slot available → increment_regen.
    Fail + cap reached → escalation (forced)."""
    if state.get("grounding_pass"):
        return "escalation"
    if state.get("grounding_regen_count", 0) < GROUNDING_MAX_REGENERATIONS:
        return "increment_regen"
    return "escalation"


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def build_graph(vectorstore_dir: Path | None = None):
    """Build and compile the main agent StateGraph.

    Args:
        vectorstore_dir: Override the ChromaDB path (useful in tests).

    Returns:
        Compiled LangGraph ready for .invoke().
    """
    rag_subgraph = build_rag_subgraph(vectorstore_dir=vectorstore_dir or VECTORSTORE_DIR)

    def _run_rag(state: dict) -> dict:
        out = rag_subgraph.invoke({
            "query":         state.get("query", ""),
            "top_k":         RAG_TOP_K,
            "rewrite_count": 0,
        })
        # Preserve existing tool_results (e.g. compat checker output from a prior node).
        # Only reset them when RAG is the first tool called (no prior tool results).
        existing_tool_results = state.get("tool_results", [])
        return {"rag_results": out.get("rag_results", []), "tool_results": existing_tool_results}

    graph = StateGraph(AgentState)

    graph.add_node("triage",             triage_node)
    graph.add_node("run_log_analyzer",   _run_log_analyzer)
    graph.add_node("run_rag",            _run_rag)
    graph.add_node("run_compat_checker", _run_compat_checker)
    graph.add_node("sufficiency",        sufficiency_node)
    graph.add_node("synthesizer",        synthesizer_node)
    graph.add_node("grounding",          grounding_node)
    graph.add_node("increment_regen",    _increment_grounding_regen)
    graph.add_node("escalation",         escalation_node)

    graph.add_edge(START, "triage")
    graph.add_conditional_edges(
        "triage", _route_from_triage,
        {"run_log_analyzer":   "run_log_analyzer",
         "run_rag":            "run_rag",
         "run_compat_checker": "run_compat_checker"},
    )
    graph.add_edge("run_log_analyzer",   "sufficiency")
    graph.add_edge("run_rag",            "sufficiency")
    graph.add_edge("run_compat_checker", "sufficiency")
    graph.add_conditional_edges(
        "sufficiency", _route_from_sufficiency,
        {"synthesizer":      "synthesizer",
         "run_log_analyzer": "run_log_analyzer",
         "run_rag":          "run_rag"},
    )
    graph.add_edge("synthesizer", "grounding")
    graph.add_conditional_edges(
        "grounding", _route_from_grounding,
        {"escalation":    "escalation",
         "increment_regen": "increment_regen"},
    )
    graph.add_edge("increment_regen", "synthesizer")
    graph.add_edge("escalation",      END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Module-level graph (lazy) + public entry point
# ---------------------------------------------------------------------------

_graph = None


def _get_graph():
    """Return the compiled graph, building it on first call."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_agent(query: str, persona: str = "support") -> dict:
    """Run the full agent pipeline and return the final state.

    Wraps graph.invoke() in an MLflow run (experiment: radvision/agent).
    All node-level metrics are logged automatically via log_metrics_safe().

    Args:
        query:   User query string.
        persona: "support" | "sales"

    Returns:
        Final AgentState dict with outcome, final_response, and all
        intermediate fields populated.
    """
    set_experiment(EXPERIMENT_AGENT)
    with mlflow.start_run():
        log_params_safe({
            "query":     query[:200],
            "persona":   persona,
            "llm_model": LLM_MODEL,
        })

        result: dict = _get_graph().invoke({
            "query":               query,
            "persona":             persona,
            "retry_count":         0,
            "grounding_regen_count": 0,
        })

        log_metrics_safe({
            "grounding_score": result.get("grounding_score", 0.0),
            "retry_count":     result.get("retry_count", 0),
        })
        set_tags_safe({"outcome": result.get("outcome", "unknown")})

    return result
