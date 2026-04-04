"""Streamlit UI for the RadVision Pro Support Agent.

Run from the project root:
    streamlit run src/ui/app.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of working directory.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import mlflow
import streamlit as st

from src.agents.graph import _get_graph
from src.config import LLM_MODEL
from src.tracking import (
    EXPERIMENT_AGENT,
    log_metrics_safe,
    log_params_safe,
    set_experiment,
    set_tags_safe,
)

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RadVision Pro Support Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("RadVision Pro")
    st.caption("Agentic RAG Support Agent")
    st.divider()

    persona = st.selectbox(
        "Persona",
        options=["support", "sales"],
        format_func=lambda p: {
            "support": "Support Engineer",
            "sales":   "Sales Engineer",
        }[p],
    )

    st.divider()
    if st.button("🗑 Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Agent runner — streams the graph and captures the execution trace
# ---------------------------------------------------------------------------


def _run_with_trace(query: str, persona: str) -> tuple[dict, list[str], str]:
    """Invoke the graph, capture per-node trace, log to MLflow.

    Returns:
        (final_state, node_trace, mlflow_run_id)
    """
    graph = _get_graph()
    node_trace: list[str] = []
    state: dict = {
        "query":               query,
        "persona":             persona,
        "retry_count":         0,
        "grounding_regen_count": 0,
    }

    set_experiment(EXPERIMENT_AGENT)
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        log_params_safe({"query": query[:200], "persona": persona, "llm_model": LLM_MODEL})

        for chunk in graph.stream(state, stream_mode="updates"):
            for node_name, updates in chunk.items():
                node_trace.append(node_name)
                state.update(updates)

        log_metrics_safe({
            "grounding_score": state.get("grounding_score", 0.0),
            "retry_count":     state.get("retry_count", 0),
        })
        set_tags_safe({"outcome": state.get("outcome", "unknown")})

    return state, node_trace, run_id


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_OUTCOME_BADGE = {
    "resolved": "🟢 Resolved",
    "clarify":  "🟡 Clarify",
    "escalate": "🔴 Escalate",
}

_ROUTE_LABEL = {
    "error_pattern": "Log Analyzer",
    "config_check":  "Compat Checker",
    "docs_kb":       "RAG / Docs KB",
}


def _render_message(msg: dict) -> None:
    """Render a stored message with all its expandable detail sections."""
    s = msg["state"]
    trace = msg["trace"]

    st.markdown(s.get("final_response", ""))

    # ---- Execution trace ----
    trace_str = " → ".join(["START"] + trace + ["END"])
    with st.expander("🗺 Execution trace"):
        st.code(trace_str, language=None)
        st.caption(f"MLflow run: `{msg['run_id']}`")

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("🔍 Triage decision"):
            st.markdown(f"**Intent:** `{s.get('intent', '—')}`")
            st.markdown(f"**Route:** `{_ROUTE_LABEL.get(s.get('route_decision', ''), s.get('route_decision', '—'))}`")
            entities = s.get("entities") or {}
            if entities:
                for k, v in entities.items():
                    st.markdown(f"- **{k}:** `{v}`")
            else:
                st.caption("No entities extracted.")

        with st.expander("✅ Sufficiency"):
            sufficient = s.get("evidence_sufficient")
            icon = "🟢" if sufficient else "🔴"
            st.markdown(f"{icon} **Sufficient:** `{sufficient}`")
            st.markdown(f"**Retry count:** `{s.get('retry_count', 0)}`")
            reason = s.get("retry_reason", "")
            if reason:
                st.caption(f"Reason: {reason}")

    with col2:
        with st.expander("🔎 Evidence gathered"):
            tool_results = s.get("tool_results") or []
            rag_results  = s.get("rag_results")  or []

            if tool_results:
                st.markdown("**Tool results**")
                for r in tool_results:
                    if r.get("matched"):
                        st.success(f"Match: {r.get('kb_ref')} — {r.get('description', '')[:80]}")
                    elif "status" in r:
                        color = "success" if r["status"] == "supported" else \
                                "warning" if r["status"] == "limited" else "error"
                        getattr(st, color)(f"{r['status'].upper()}: {r.get('reason', '')[:80]}")
                    else:
                        st.info("No match found.")

            if rag_results:
                st.markdown("**RAG results**")
                for r in rag_results[:3]:
                    meta = r.get("metadata", {})
                    label = meta.get("kb_id") or meta.get("ticket_id") or r.get("source_collection", "")
                    st.markdown(
                        f"`{label}` — score `{r['score']:.3f}` "
                        f"({r.get('source_collection', '')})"
                    )
                    st.caption(r.get("text", "")[:120])

            if not tool_results and not rag_results:
                st.caption("No evidence gathered.")

        with st.expander("🔗 Grounding"):
            score = s.get("grounding_score", 0.0)
            passed = s.get("grounding_pass", False)
            icon = "🟢" if passed else "🔴"
            st.markdown(f"{icon} **Score:** `{score:.2f}` &nbsp; pass=`{passed}`")
            st.markdown(f"**Regen attempts:** `{s.get('grounding_regen_count', 0)}`")
            ungrounded = s.get("ungrounded_claims") or []
            if ungrounded:
                st.warning("Ungrounded claims:")
                for claim in ungrounded:
                    st.caption(f"• {claim}")

    with st.expander(f"📊 Outcome — {_OUTCOME_BADGE.get(s.get('outcome', ''), s.get('outcome', ''))}"):
        outcome = s.get("outcome", "")
        if outcome == "escalate" and s.get("escalation_package"):
            pkg = s["escalation_package"]
            st.error(f"Escalation reason: {pkg.get('escalation_reason', '')}")
        elif outcome == "clarify":
            st.warning(s.get("clarification_question", ""))
        else:
            st.success("Query resolved.")


# ---------------------------------------------------------------------------
# Replay stored messages
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message("user"):
        st.markdown(msg["query"])
    with st.chat_message("assistant"):
        _render_message(msg)

# ---------------------------------------------------------------------------
# Handle new input
# ---------------------------------------------------------------------------

if query := st.chat_input("Ask about RadVision Pro..."):
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Running agent…"):
            try:
                state, trace, run_id = _run_with_trace(query, persona)
            except Exception as exc:
                st.error(f"Agent error: {exc}")
                st.stop()

        msg = {"query": query, "state": state, "trace": trace, "run_id": run_id}
        st.session_state.messages.append(msg)
        _render_message(msg)
