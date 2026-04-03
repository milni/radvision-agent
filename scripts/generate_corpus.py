"""Generate synthetic corpus documents from the product spec seed YAML.

Reads data/seed/product_spec.yaml and writes markdown files into data/corpus/:

  kb_articles/   — Knowledge Base (KB) articles: official, curated troubleshooting
                   documents written after a problem is fully understood.
                   Each article covers one known issue (KB-XXXX.md).

  product_docs/  — Component reference docs with configuration settings.

  release_notes/ — Per-version changelogs listing new features and known issues.

  past_tickets/  — Raw support case records from customer incidents. Unlike KB
                   articles (which give the clean, canonical answer), tickets
                   capture the engineer's diagnosis steps, environment details,
                   and field experience that may not appear in a KB article.

Together, KB articles and past tickets give the RAG agent both the official fix
and real-world field context — which is why they are separate collections with
different chunking strategies in Phase 3.

The markdown structure is intentional: Phase 3 uses different chunking
strategies per collection (by section heading, by paragraph, etc.).
Run from the project root:

    python scripts/generate_corpus.py
"""

import sys
from pathlib import Path

import yaml

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import CORPUS_DIR, SEED_PATH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(name: str) -> str:
    """Convert a component display name to a lowercase underscore filename slug.

    Used as a fallback when a component name is not in _COMPONENT_SLUGS.
    E.g. "3D Rendering Engine" → "rendering_engine".
    """
    return (
        name.lower()
        .replace("3d ", "")
        .replace(" ", "_")
        .replace("-", "_")
    )


def _write(path: Path, content: str) -> None:
    """Create parent directories if needed and write content to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# KB Articles
# Each file has distinct ## sections so Phase 3 can chunk by heading.
# ---------------------------------------------------------------------------

def generate_kb_articles(spec: dict) -> list[Path]:
    """Generate one markdown Knowledge Base (KB) article per known issue across all versions.

    KB articles are the official, curated answer to a known product problem.
    They are identified by a KB-XXXX ID and cover a single issue with a clean,
    prescriptive structure. Contrast with past tickets (generate_past_tickets),
    which are raw case records capturing field experience and diagnosis steps
    that may not appear in the KB article.

    Iterates every version in the spec and writes a file for each known_issue
    entry. The heading structure (Symptom / Root Cause / Workaround / Resolution)
    is required by Phase 3, which chunks KB articles by section.

    Args:
        spec: Parsed product_spec.yaml as a dict.

    Returns:
        List of paths to the written files.
    """
    out_dir = CORPUS_DIR / "kb_articles"
    written: list[Path] = []

    # Collect every known_issue across all versions, noting the affected version
    for version in spec.get("versions", []):
        ver = version["version"]
        for issue in version.get("known_issues", []):
            kb_id = issue["id"]
            components = ", ".join(issue.get("affected_components", []))
            fixed = issue.get("fixed_in") or "Not yet fixed"

            content = f"""\
# {kb_id}: {issue['title']}

**Article ID:** {kb_id}
**Affected Version:** {ver}
**Affected Components:** {components}
**Fixed In:** {fixed}

## Symptom

{issue['symptom']}

## Root Cause

{issue['root_cause']}

## Workaround

{issue['workaround']}

## Resolution

{"This issue is resolved in version " + fixed + "." if issue.get('fixed_in') else "No permanent fix available yet. Apply the workaround above."}
"""
            path = out_dir / f"{kb_id}.md"
            _write(path, content)
            written.append(path)

    return written


# ---------------------------------------------------------------------------
# Product Docs
# Hierarchical headings so Phase 3 can prepend heading context per paragraph.
# ---------------------------------------------------------------------------

_COMPONENT_SLUGS = {
    "DICOM Gateway": "dicom_gateway",
    "3D Rendering Engine": "rendering_engine",
    "Integration Gateway": "integration_gateway",
    "Collaboration Hub": "collab_hub",
}


def generate_product_docs(spec: dict) -> list[Path]:
    """Generate one markdown reference doc per product component.

    Each file covers the component overview, configuration file path, and a
    per-setting section (### SETTING_NAME). The hierarchical heading structure
    allows Phase 3 to prepend heading context when chunking by paragraph.

    Args:
        spec: Parsed product_spec.yaml as a dict.

    Returns:
        List of paths to the written files.
    """
    out_dir = CORPUS_DIR / "product_docs"
    written: list[Path] = []

    for component in spec.get("components", []):
        name = component["name"]
        slug = _COMPONENT_SLUGS.get(name, _slug(name))

        # Protocols or features line
        extras = component.get("protocols") or component.get("features") or []
        extras_line = f"**{'Protocols' if 'protocols' in component else 'Features'}:** {', '.join(extras)}\n" if extras else ""

        # Build configuration settings table
        settings_lines = []
        for s in component.get("key_settings", []):
            settings_lines.append(f"### {s['name']}\n\n- **Default:** `{s['default']}`\n- **Description:** {s['description']}\n")

        settings_section = "\n".join(settings_lines)

        content = f"""\
# {name}

## Overview

{component['description']}

**Configuration file:** `{component['config_file']}`
{extras_line}
## Configuration Reference

{settings_section}
"""
        path = out_dir / f"{slug}.md"
        _write(path, content)
        written.append(path)

    return written



# ---------------------------------------------------------------------------
# Release Notes
# One file per version; known issues listed inline for cross-referencing.
# ---------------------------------------------------------------------------

def generate_release_notes(spec: dict) -> list[Path]:
    """Generate one markdown release notes file per product version.

    Each file lists the highlights (What's New) and any known issues for that
    version, including inline workarounds. Known issues are cross-referenced by
    KB ID so the release notes and KB articles stay consistent.

    Args:
        spec: Parsed product_spec.yaml as a dict.

    Returns:
        List of paths to the written files.
    """
    out_dir = CORPUS_DIR / "release_notes"
    written: list[Path] = []

    for version in spec.get("versions", []):
        ver = version["version"]
        slug = f"v{ver.replace('.', '_')}"

        highlights = "\n".join(f"- {h}" for h in version.get("highlights", []))

        issues = version.get("known_issues", [])
        if issues:
            issue_lines = []
            for issue in issues:
                fixed = issue.get("fixed_in") or "open"
                issue_lines.append(
                    f"### {issue['id']}: {issue['title']}\n\n"
                    f"**Components:** {', '.join(issue.get('affected_components', []))}  \n"
                    f"**Status:** {'Fixed in ' + fixed if issue.get('fixed_in') else 'Open — see workaround'}  \n\n"
                    f"{issue['symptom']}\n\n"
                    f"**Workaround:** {issue['workaround']}\n"
                )
            known_issues_section = "## Known Issues\n\n" + "\n".join(issue_lines)
        else:
            known_issues_section = "## Known Issues\n\nNo known issues for this release.\n"

        content = f"""\
# RadVision Pro v{ver} Release Notes

**Release Date:** {version['release_date']}
**Status:** {version['status'].replace('_', ' ').title()}

## What's New

{highlights}

{known_issues_section}
"""
        path = out_dir / f"{slug}.md"
        _write(path, content)
        written.append(path)

    return written


# ---------------------------------------------------------------------------
# Past Tickets
# Resolution section is prominent for Phase 3 chunking.
# ---------------------------------------------------------------------------

def generate_past_tickets(spec: dict) -> list[Path]:
    """Generate one markdown file per past support ticket template.

    Past tickets are raw case records from real customer incidents. Unlike KB
    articles (generate_kb_articles), which give the clean canonical answer,
    tickets capture the engineer's diagnosis steps, environment specifics, and
    field observations. They complement KB articles by providing real-world
    context: e.g. a ticket may note that framerate also needed reducing, or
    that a hardware upgrade was recommended — details absent from the KB article.

    Each file captures the customer environment, symptom, step-by-step diagnosis,
    and resolution. The prominent ## Resolution section is the primary chunk target
    for Phase 3, since it contains the actionable fix.

    Args:
        spec: Parsed product_spec.yaml as a dict.

    Returns:
        List of paths to the written files.
    """
    out_dir = CORPUS_DIR / "past_tickets"
    written: list[Path] = []

    for ticket in spec.get("ticket_templates", []):
        tid = ticket["ticket_id"]
        env = ticket["environment"]
        persona = ticket["persona_filed_by"].replace("_", " ").title()

        steps = "\n".join(f"{i+1}. {s}" for i, s in enumerate(ticket.get("diagnosis_steps", [])))
        related = ", ".join(ticket.get("related_kb", [])) or "None"

        content = f"""\
# {tid}: {ticket['title']}

**Customer:** {ticket['customer']}
**Filed by:** {persona}
**Environment:** RadVision Pro v{env['version']}, {env['os']}, {env['gpu']}

## Symptom

{ticket['symptom']}

## Diagnosis Steps

{steps}

## Resolution

{ticket['resolution']}

**Resolution time:** {ticket['resolution_time_hours']} hours
**Related KB articles:** {related}
"""
        path = out_dir / f"{tid}.md"
        _write(path, content)
        written.append(path)

    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all four generators and print a summary of written files."""
    spec = yaml.safe_load(SEED_PATH.read_text())

    kb = generate_kb_articles(spec)
    docs = generate_product_docs(spec)
    notes = generate_release_notes(spec)
    tickets = generate_past_tickets(spec)

    all_files = kb + docs + notes + tickets
    print(f"Generated {len(all_files)} files into {CORPUS_DIR}/\n")
    for f in all_files:
        rel = f.relative_to(CORPUS_DIR.parent.parent)
        print(f"  {rel}")


if __name__ == "__main__":
    main()
