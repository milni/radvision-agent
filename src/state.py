"""LangGraph state definition for the RadVision Pro support agent.

This TypedDict defines all fields that flow through the agent graph.
Each node reads from and writes to this shared state.
"""

from typing import TypedDict


class AgentState(TypedDict):
    """Shared state for the RadVision Pro support agent graph.

    Fields are populated progressively as the query flows through nodes.
    Downstream nodes should check for None/empty before using upstream results.
    """

    # --- Input ---
    query: str
    persona: str  # "sales" | "support"

    # --- Node 1: Triage + Route ---
    entities: dict  # version, os, gpu, error_code, component, site, etc.
    intent: str  # "troubleshooting" | "feature_inquiry" | "config_help" | "general"
    route_decision: str  # "error_pattern" | "docs_kb" | "config_check"

    # --- Tool / RAG results ---
    tool_results: list[dict]  # Outputs from log_analyzer and/or compat_checker
    rag_results: list[dict]  # Retrieved + reranked chunks with metadata

    # --- Node 2: Sufficiency Check ---
    evidence_sufficient: bool
    retry_count: int  # Tracks retries, max 1
    retry_reason: str  # Why evidence was insufficient

    # --- Node 3: Response Synthesizer ---
    draft_response: str  # Persona-shaped response draft

    # --- Node 4: Grounding Checker ---
    grounding_score: float  # 0.0–1.0
    ungrounded_claims: list[str]  # Claims not supported by evidence
    grounding_pass: bool  # True if score above threshold
    grounding_regen_count: int  # Tracks regeneration attempts, max GROUNDING_MAX_REGENERATIONS

    # --- Node 5: Escalation Gate ---
    outcome: str  # "resolved" | "clarify" | "escalate"
    final_response: str  # Final output to user
    clarification_question: str  # If outcome == "clarify"
    escalation_package: dict | None  # If outcome == "escalate"
