"""Response Synthesizer node (Node 3) for the RadVision Pro support agent.

Takes the gathered evidence (tool_results + rag_results) and drafts a
response shaped to the caller's persona via an LLM call (gemma3:4b by default).

Personas:
  field_engineer — technical, direct; exact settings, config paths, commands
  support        — step-by-step; numbered resolution steps, KB references
  sales          — feature-focused; no raw technical detail, upgrade framing
"""

import logging

from langchain_ollama import ChatOllama

from src.config import LLM_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

_MAX_EVIDENCE_CHARS = 1200

_PERSONA_INSTRUCTION = {
    "field_engineer": (
        "You are assisting a field engineer. Be technical and direct. "
        "Include exact config file paths, setting names, and commands. "
        "Reference KB articles and ticket IDs where relevant."
    ),
    "support": (
        "You are assisting a support engineer. Provide numbered resolution steps. "
        "Include KB article references. End with a confirmation prompt."
    ),
    "sales": (
        "You are assisting a sales engineer. Use customer-friendly language. "
        "Avoid raw config details. Frame limitations as upgrade opportunities."
    ),
}

_SYNTH_PROMPT = """\
{persona_instruction}

Using ONLY the evidence below, write a concise support response.
Do not invent facts not present in the evidence.

Evidence:
{evidence}

References: {refs}

Query: {query}

Response:"""

_NO_EVIDENCE_MSG = {
    "field_engineer": (
        "No matching documentation or knowledge base article was found for this query. "
        "Please check the system logs and escalate with full diagnostic output."
    ),
    "sales": (
        "I don't have specific documentation on this topic in our current knowledge base. "
        "I'll connect you with our technical team who can provide accurate guidance."
    ),
    "support": (
        "No relevant information was found for this query in the knowledge base. "
        "Please verify the issue details and consider escalating to the engineering team."
    ),
}

_llm = None


def _get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    return _llm


def _best_evidence_text(tool_results: list[dict], rag_results: list[dict]) -> str:
    for r in tool_results:
        if r.get("matched") and r.get("description"):
            return (
                f"Issue: {r['description']}\n"
                f"Component: {r.get('component', 'unknown')}\n"
                f"Severity: {r.get('severity', 'unknown')}\n"
                f"Reference: {r.get('kb_ref', '')}"
            )
    if rag_results:
        return rag_results[0]["text"][:_MAX_EVIDENCE_CHARS]
    return ""


def _kb_refs(tool_results: list[dict], rag_results: list[dict]) -> list[str]:
    refs: list[str] = []
    for r in tool_results:
        if r.get("kb_ref"):
            refs.append(r["kb_ref"])
    for r in rag_results:
        kb = r.get("metadata", {}).get("kb_id")
        if kb:
            refs.append(kb)
        tid = r.get("metadata", {}).get("ticket_id")
        if tid:
            refs.append(tid)
    seen: set = set()
    return [x for x in refs if not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]


def synthesizer_node(state: dict) -> dict:
    """Draft a response from the evidence, shaped by the caller's persona."""
    persona:      str        = state.get("persona", "support")
    tool_results: list[dict] = state.get("tool_results", [])
    rag_results:  list[dict] = state.get("rag_results", [])
    query:        str        = state.get("query", "")

    evidence = _best_evidence_text(tool_results, rag_results)

    if not evidence:
        draft = _NO_EVIDENCE_MSG.get(persona, _NO_EVIDENCE_MSG["support"])
        logger.info("Synthesizer: no evidence — returning fallback for persona=%s", persona)
        return {"draft_response": draft}

    refs = _kb_refs(tool_results, rag_results)
    prompt = _SYNTH_PROMPT.format(
        persona_instruction=_PERSONA_INSTRUCTION.get(persona, _PERSONA_INSTRUCTION["support"]),
        evidence=evidence,
        refs=", ".join(refs) if refs else "none",
        query=query,
    )
    response = _get_llm().invoke(prompt)
    draft = response.content.strip()

    logger.info("Synthesizer: persona=%s refs=%s", persona, refs)
    return {"draft_response": draft}
