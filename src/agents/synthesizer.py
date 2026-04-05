"""Response Synthesizer node (Node 3) for the RadVision Pro support agent.

Takes the gathered evidence (tool_results + rag_results) and drafts a
response shaped to the caller's persona via an LLM call (gemma3:4b by default).

Personas:
  support — step-by-step; numbered resolution steps, KB references
  sales   — feature-focused; no raw technical detail, upgrade framing
"""

import logging

import chromadb
from langchain_ollama import ChatOllama

from src.config import LLM_MODEL, OLLAMA_BASE_URL, RAG_RELEVANCE_THRESHOLD, VECTORSTORE_DIR

logger = logging.getLogger(__name__)

_MAX_EVIDENCE_CHARS = 1200

_PERSONA_INSTRUCTION = {
    "support": (
        "You are assisting a support engineer. Provide numbered resolution steps. "
        "Include KB article references. "
        "If 'KB Workaround' or 'KB Resolution' blocks are present in the evidence, "
        "include them verbatim in your response under matching headings. "
        "End with a confirmation prompt asking whether the issue is resolved."
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
Do not start with preambles like "Okay", "Sure", "Here's", "Subject:", or "Thank you for contacting". Begin directly with the content.

Evidence:
{evidence}
{resolution_block}
References: {refs}

Query: {query}

Response:"""

_NO_EVIDENCE_PROMPT = {
    "sales": (
        "You are assisting a sales engineer for RadVision Pro, a radiology 2D/3D visualization platform.\n"
        "The customer asked: {query}\n\n"
        "This topic or feature does not appear to be part of the current RadVision Pro product.\n"
        "Write 1-2 sentences: acknowledge it is not a current feature, and offer to connect them with "
        "the product team to discuss their use case and the roadmap. "
        "Use customer-friendly language. Do not invent features."
    ),
    "support": (
        "You are assisting a support engineer for RadVision Pro, a radiology 2D/3D visualization platform.\n"
        "The user asked: {query}\n\n"
        "No relevant information was found in the knowledge base for this query. "
        "This may be because the query is too vague, or the topic is not covered.\n"
        "Write 2-3 sentences. Do NOT invent resolution steps or technical details. "
        "Ask the user to clarify: which version of RadVision Pro, which component or feature is affected, "
        "and what specific symptoms or error messages they are seeing. "
        "Be concise and professional."
    ),
}

_llm = None
_chroma = None


def _get_chroma() -> chromadb.PersistentClient:
    global _chroma
    if _chroma is None:
        _chroma = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    return _chroma


def _fetch_kb_sections(
    kb_ids: list[str],
    sections: list[str],
) -> dict[str, dict[str, str]]:
    """Return {kb_id: {section: text}} for the requested sections of cited KB articles.

    Looks up pre-chunked sections directly by metadata filter — no embedding search needed.
    Missing sections (article has no such section) are silently skipped.
    """
    if not kb_ids or not sections:
        return {}
    try:
        col = _get_chroma().get_collection("kb_articles")
    except Exception:
        logger.warning("Synthesizer: kb_articles collection not available for section lookup")
        return {}

    result: dict[str, dict[str, str]] = {}
    for kb_id in kb_ids:
        for section in sections:
            try:
                data = col.get(where={"$and": [{"kb_id": kb_id}, {"section": section}]})
                if data["documents"]:
                    raw = data["documents"][0]
                    marker = f"## {section}\n\n"
                    text = raw.split(marker, 1)[1].strip() if marker in raw else raw
                    result.setdefault(kb_id, {})[section] = text
            except Exception as exc:
                logger.debug("Synthesizer: could not fetch %s/%s: %s", kb_id, section, exc)
    return result


def _get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    return _llm


def _best_evidence_text(tool_results: list[dict], rag_results: list[dict]) -> str:
    # Log analyzer match
    for r in tool_results:
        if r.get("matched") and r.get("description"):
            return (
                f"Issue: {r['description']}\n"
                f"Component: {r.get('component', 'unknown')}\n"
                f"Severity: {r.get('severity', 'unknown')}\n"
                f"Reference: {r.get('kb_ref', '')}"
            )
    # Compat checker results — collect all rows with a known status
    compat_rows = [r for r in tool_results if r.get("status") in
                   ("supported", "limited", "unsupported", "experimental", "unknown")]
    if compat_rows:
        lines = []
        for r in compat_rows:
            feature  = r.get("checked_feature", r.get("feature", ""))
            version  = r.get("checked_version", r.get("version", ""))
            gpu      = r.get("gpu", "")
            status   = r.get("status", "")
            reason   = r.get("reason", "")
            rec      = r.get("recommendation", "")
            line = f"Feature: {feature} | Version: {version} | GPU: {gpu} | Status: {status}"
            if reason:
                line += f" | {reason}"
            if rec:
                line += f"\nRecommendation: {rec}"
            lines.append(line)
        return "\n\n".join(lines)[:_MAX_EVIDENCE_CHARS]
    relevant_rag = [r for r in rag_results if r.get("score", 0) >= RAG_RELEVANCE_THRESHOLD * 1.5]
    if relevant_rag:
        parts: list[str] = []
        budget = _MAX_EVIDENCE_CHARS
        for r in relevant_rag:
            chunk = r["text"]
            if len(chunk) > budget:
                parts.append(chunk[:budget])
                break
            parts.append(chunk)
            budget -= len(chunk)
        return "\n\n---\n\n".join(parts)
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
        prompt_tmpl = _NO_EVIDENCE_PROMPT.get(persona, _NO_EVIDENCE_PROMPT["support"])
        response = _get_llm().invoke(prompt_tmpl.format(query=query))
        draft = response.content.strip()
        logger.info("Synthesizer: no evidence — LLM no-evidence response for persona=%s", persona)
        return {"draft_response": draft}

    refs = _kb_refs(tool_results, rag_results)

    # For support engineers, fetch Workaround and Resolution sections of cited KB articles.
    resolution_block = ""
    if persona == "support":
        kb_ids = [r for r in refs if r.startswith("KB-")]
        kb_sections = _fetch_kb_sections(kb_ids, ["Workaround", "Resolution"])
        if kb_sections:
            parts = []
            for kb_id, sections in kb_sections.items():
                for section_name, text in sections.items():
                    parts.append(f"KB {section_name} — {kb_id}:\n{text}")
            resolution_block = "\n" + "\n\n".join(parts) + "\n"
            logger.info("Synthesizer: appended sections for %s", {k: list(v) for k, v in kb_sections.items()})

    prompt = _SYNTH_PROMPT.format(
        persona_instruction=_PERSONA_INSTRUCTION.get(persona, _PERSONA_INSTRUCTION["support"]),
        evidence=evidence,
        resolution_block=resolution_block,
        refs=", ".join(refs) if refs else "none",
        query=query,
    )
    response = _get_llm().invoke(prompt)
    draft = response.content.strip()

    logger.info("Synthesizer: persona=%s refs=%s", persona, refs)
    return {"draft_response": draft}
