"""Streamlit UI for the RadVision Pro Support Agent.

Run from the project root:
    streamlit run src/ui/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import mlflow
import pandas as pd
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
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RadVision Pro — Support Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Architecture diagram — static HTML, shown in sidebar
# ---------------------------------------------------------------------------

_ARCH_HTML = """
<style>
.arch { font-family: monospace; font-size: 0.78rem; line-height: 1.7; color: #ccc; }
.arch .node  { display:inline-block; background:#1a3a5c; color:#fff;
               padding:2px 8px; border-radius:4px; white-space:nowrap; }
.arch .node.tool  { background:#1a5c3a; }
.arch .node.check { background:#5c5c1a; }
.arch .node.synth { background:#1a5c5c; }
.arch .node.grnd  { background:#5c1a3a; }
.arch .node.esc   { background:#7a0000; }
.arch .node.se    { background:#444; }
.arch .arr  { color:#888; }
.arch .lbl  { color:#aaa; font-size:0.7rem; }
</style>
<div class="arch">
  <span class="node se">START</span><br>
  <span class="arr">&nbsp;&nbsp;↓</span><br>
  <span class="node">🔀 Triage</span><br>
  <span class="arr">&nbsp;&nbsp;↓ route decision</span><br>
  <table style="font-family:monospace;font-size:0.75rem;color:#ccc;margin-left:4px">
    <tr>
      <td class="lbl" style="padding-right:6px">error_pattern</td>
      <td>→</td>
      <td><span class="node tool">🔍 Log Analyzer</span></td>
    </tr>
    <tr>
      <td class="lbl" style="padding-right:6px">config_check&nbsp;</td>
      <td>→</td>
      <td><span class="node tool">📊 Compat Checker</span></td>
    </tr>
    <tr>
      <td class="lbl" style="padding-right:6px">docs_kb&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</td>
      <td>→</td>
      <td><span class="node tool">📚 RAG Retrieval</span></td>
    </tr>
  </table>
  <span class="arr">&nbsp;&nbsp;↓</span><br>
  <span class="node check">✅ Sufficiency</span>
  <span class="lbl"> ← retry (max 1)</span><br>
  <span class="arr">&nbsp;&nbsp;↓</span><br>
  <span class="node synth">✍️ Synthesizer</span>
  <span class="lbl"> ← regen (max 1)</span><br>
  <span class="arr">&nbsp;&nbsp;↓</span><br>
  <span class="node grnd">🔗 Grounding</span><br>
  <span class="arr">&nbsp;&nbsp;↓</span><br>
  <span class="node esc">🚨 Escalation Gate</span><br>
  <span class="arr">&nbsp;&nbsp;↓</span><br>
  <span class="node se">END</span>
</div>
"""

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🏥 RadVision Pro")
    st.caption("Agentic RAG Support Agent")
    st.divider()

    persona = st.selectbox(
        "Persona",
        options=["support", "sales"],
        format_func=lambda p: {
            "support": "🔧 Support Engineer",
            "sales":   "💼 Sales Engineer",
        }[p],
    )

    st.divider()
    st.markdown("**Agent Architecture**")
    st.html(_ARCH_HTML)

    st.divider()
    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Node display metadata
# ---------------------------------------------------------------------------

_NODE_LABEL = {
    "triage":             "🔀 Triage",
    "run_log_analyzer":   "🔍 Log Analyzer",
    "run_compat_checker": "📊 Compat Checker",
    "run_rag":            "📚 RAG",
    "sufficiency":        "✅ Sufficiency",
    "synthesizer":        "✍️ Synthesizer",
    "grounding":          "🔗 Grounding",
    "increment_regen":    "🔄 Regen",
    "escalation":         "🚨 Escalation",
}

_NODE_COLOR = {
    "triage":             "#1a3a5c",
    "run_log_analyzer":   "#1a5c3a",
    "run_compat_checker": "#1a5c3a",
    "run_rag":            "#3a1a5c",
    "sufficiency":        "#5c5c1a",
    "synthesizer":        "#1a5c5c",
    "grounding":          "#5c1a3a",
    "increment_regen":    "#5c3a1a",
    "escalation":         "#7a0000",
}

# ---------------------------------------------------------------------------
# Pipeline renderer — always visible, no click required
# ---------------------------------------------------------------------------


def _pipeline_html(trace: list[str]) -> str:
    nodes = ["START"] + trace + ["END"]
    parts = []
    for i, n in enumerate(nodes):
        color = _NODE_COLOR.get(n, "#444")
        label = _NODE_LABEL.get(n, n.upper())
        parts.append(
            f'<span style="background:{color};color:#fff;padding:4px 10px;'
            f'border-radius:6px;font-size:0.75rem;white-space:nowrap">{label}</span>'
        )
        if i < len(nodes) - 1:
            parts.append('<span style="color:#888;padding:0 3px">→</span>')
    return (
        '<div style="display:flex;flex-wrap:wrap;align-items:center;'
        'gap:4px;padding:6px 0">' + "".join(parts) + "</div>"
    )


# ---------------------------------------------------------------------------
# Sources renderer — shown when user clicks the expander
# ---------------------------------------------------------------------------


def _render_sources(s: dict, key: str) -> None:
    """Show the evidence that was passed to the Sufficiency node."""
    tool_results = s.get("tool_results") or []
    rag_results  = s.get("rag_results")  or []

    if not tool_results and not rag_results:
        st.caption("No sources were gathered.")
        return

    # Log analyzer results
    log_hits = [r for r in tool_results if "matched" in r]
    if log_hits:
        st.markdown("**Log Analyzer**")
        for r in log_hits:
            if r.get("matched"):
                st.success(f"✓ Matched **{r.get('kb_ref')}** — {r.get('description', '')}")
            else:
                st.info("No pattern matched in log analyzer.")

    # Compat checker results — styled dataframe
    compat_rows = [r for r in tool_results if "status" in r and "matched" not in r]
    if compat_rows:
        st.markdown("**Compatibility Checker**")
        df_data = []
        for r in compat_rows:
            df_data.append({
                "Feature": r.get("checked_feature", ""),
                "Version": r.get("checked_version", ""),
                "Status":  r.get("status", "").upper(),
                "Detail":  r.get("reason", "")[:80],
            })
        if df_data:
            df = pd.DataFrame(df_data)

            def _color(val):
                return {
                    "SUPPORTED":    "background-color:#d4edda;color:#155724",
                    "LIMITED":      "background-color:#fff3cd;color:#856404",
                    "UNSUPPORTED":  "background-color:#f8d7da;color:#721c24",
                    "EXPERIMENTAL": "background-color:#d1ecf1;color:#0c5460",
                    "UNKNOWN":      "background-color:#e2e3e5;color:#383d41",
                }.get(val, "")

            st.dataframe(
                df.style.map(_color, subset=["Status"]),
                hide_index=True,
                use_container_width=True,
            )

    # RAG chunks
    if rag_results:
        st.markdown("**Retrieved Chunks**")
        for r in rag_results[:5]:
            meta    = r.get("metadata", {})
            score   = r.get("score", 0.0)
            label   = meta.get("kb_id") or meta.get("source_file", "") or r.get("source_collection", "")
            section = meta.get("section", "")
            header  = f"`{label}`" + (f" › `{section}`" if section else "")
            col_l, col_r = st.columns([4, 1])
            with col_l:
                st.markdown(header)
                st.markdown(r.get("text", ""))
            with col_r:
                st.progress(min(score, 1.0), text=f"{score:.2f}")


# ---------------------------------------------------------------------------
# Agent runner with live streaming
# ---------------------------------------------------------------------------


def _run_with_trace(query: str, persona: str) -> tuple[dict, list[str], str]:
    graph = _get_graph()
    node_trace: list[str] = []
    state: dict = {
        "query":                 query,
        "persona":               persona,
        "retry_count":           0,
        "grounding_regen_count": 0,
    }

    set_experiment(EXPERIMENT_AGENT)
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        log_params_safe({"query": query[:200], "persona": persona, "llm_model": LLM_MODEL})

        with st.status("Running…", expanded=True) as status:
            for chunk in graph.stream(state, stream_mode="updates"):
                for node_name, updates in chunk.items():
                    node_trace.append(node_name)
                    state.update(updates)
                    st.write(_NODE_LABEL.get(node_name, node_name))
            status.update(label="Done", state="complete", expanded=False)

        log_metrics_safe({
            "grounding_score": state.get("grounding_score", 0.0),
            "retry_count":     state.get("retry_count", 0),
        })
        set_tags_safe({"outcome": state.get("outcome", "unknown")})

    return state, node_trace, run_id


# ---------------------------------------------------------------------------
# Per-message renderer
# ---------------------------------------------------------------------------


def _render_message(msg: dict) -> None:
    s     = msg["state"]
    trace = msg["trace"]

    # 1. Execution pipeline — always visible
    st.html(_pipeline_html(trace))

    # 2. Answer
    outcome = s.get("outcome", "")
    if outcome == "escalate":
        st.error(s.get("final_response", ""))
    else:
        st.markdown(s.get("final_response", ""))

    # 3. Sources — one click to expand; key scoped to this message to prevent duplication
    with st.expander("📎 Sources", key=f"sources_{msg['run_id']}"):
        _render_sources(s, key=msg["run_id"])


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

st.title("🏥 RadVision Pro Support Agent")
st.divider()

for msg in st.session_state.messages:
    with st.chat_message("user"):
        st.markdown(msg["query"])
    with st.chat_message("assistant"):
        _render_message(msg)

if query := st.chat_input("Ask about RadVision Pro…"):
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            state, trace, run_id = _run_with_trace(query, persona)
        except Exception as exc:
            st.error(f"Agent error: {exc}")
            st.stop()

        msg = {"query": query, "state": state, "trace": trace, "run_id": run_id}
        st.session_state.messages.append(msg)
        _render_message(msg)
