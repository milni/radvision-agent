"""Response Synthesizer node (Node 3) for the RadVision Pro support agent.

Takes the gathered evidence (tool_results + rag_results) and drafts a
response shaped to the caller's persona.

Persona templates (dummy LLM):
  field_engineer — technical, direct; exact settings, config paths, commands
  support        — step-by-step; numbered resolution steps, KB references
  sales          — feature-focused; no raw technical detail, upgrade framing

Real LLM swap: replace _dummy_synthesize() with a structured prompt call
that receives the same evidence summary and persona instruction.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Maximum characters of evidence text to include in the response
_MAX_EVIDENCE_CHARS = 600


def _best_evidence_text(tool_results: list[dict], rag_results: list[dict]) -> str:
    """Return the most relevant evidence text available, or None if nothing found."""
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


def _no_evidence_response(persona: str) -> str:
    """Consistent fallback when all evidence sources returned nothing."""
    if persona == "field_engineer":
        return (
            "No matching documentation or knowledge base article was found for this query. "
            "Please check the system logs and escalate with full diagnostic output."
        )
    if persona == "sales":
        return (
            "I don't have specific documentation on this topic in our current knowledge base. "
            "I'll connect you with our technical team who can provide accurate guidance."
        )
    # support (default)
    return (
        "No relevant information was found for this query in the knowledge base. "
        "Please verify the issue details and consider escalating to the engineering team."
    )


def _kb_refs(tool_results: list[dict], rag_results: list[dict]) -> list[str]:
    """Collect all KB/ticket IDs mentioned in the evidence."""
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
    # preserve order, deduplicate
    seen: set = set()
    return [x for x in refs if not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]


def _format_field_engineer(evidence: str, refs: list[str], version: str) -> str:
    """Format the evidence as a technical brief for a field engineer.

    Includes the full evidence text, affected version, and KB/ticket references.
    Ends with an escalation prompt if the fix doesn't resolve the issue.
    """
    ref_line = f"\n**References:** {', '.join(refs)}" if refs else ""
    ver_line = f"\n**Affected version:** {version}" if version else ""
    return (
        f"**Technical details**\n\n"
        f"{evidence}"
        f"{ver_line}"
        f"{ref_line}\n\n"
        f"Apply the workaround above. If the issue persists, escalate with full logs."
    )


def _format_support(evidence: str, refs: list[str], version: str) -> str:
    """Format the evidence as numbered resolution steps for a support engineer.

    Splits the evidence into lines and numbers up to 6 steps. Appends KB/ticket
    references and a confirmation prompt at the end.
    """
    # Turn evidence lines into numbered steps
    lines = [l.strip() for l in evidence.split("\n") if l.strip()]
    steps = "\n".join(f"{i}. {line}" for i, line in enumerate(lines[:6], 1))
    ref_line = f"\n\n**References:** {', '.join(refs)}" if refs else ""
    return (
        f"**Resolution steps**\n\n"
        f"{steps}"
        f"{ref_line}\n\n"
        f"Follow the steps above and confirm resolution with the customer."
    )


def _format_sales(evidence: str, refs: list[str], version: str) -> str:
    """Format the evidence as a customer-friendly message for a sales engineer.

    Strips technical detail: uses only the first sentence of the evidence,
    removes markdown formatting, and frames the response around upgrade guidance
    rather than raw fix instructions.
    """
    # Extract first sentence as a plain-language summary
    first = re.split(r"[.!?\n]", evidence)[0].strip()
    # Strip markdown formatting
    first = re.sub(r"\*+|`", "", first).strip()
    ver_line = (
        f" This was addressed in version {version}." if version else ""
    )
    return (
        f"Our team is aware of this topic and has documented guidance available.{ver_line} "
        f"{first}. "
        f"I'd recommend reaching out to our support team to discuss the best upgrade path "
        f"and ensure your environment is on the latest release."
    )


def synthesizer_node(state: dict) -> dict:
    """Draft a response from the evidence, shaped by the caller's persona."""
    persona:      str        = state.get("persona", "support")
    tool_results: list[dict] = state.get("tool_results", [])
    rag_results:  list[dict] = state.get("rag_results", [])
    entities:     dict       = state.get("entities", {})
    version:      str        = entities.get("version", "")

    evidence = _best_evidence_text(tool_results, rag_results)

    if not evidence:
        # Nothing was found after all retries — produce a clean "no info" message
        # rather than formatting the fallback string through persona templates.
        draft = _no_evidence_response(persona)
        logger.info("Synthesizer: no evidence — returning fallback for persona=%s", persona)
        return {"draft_response": draft}

    refs = _kb_refs(tool_results, rag_results)

    if persona == "field_engineer":
        draft = _format_field_engineer(evidence, refs, version)
    elif persona == "sales":
        draft = _format_sales(evidence, refs, version)
    else:  # support (default)
        draft = _format_support(evidence, refs, version)

    logger.info("Synthesizer: persona=%s refs=%s", persona, refs)
    return {"draft_response": draft}
