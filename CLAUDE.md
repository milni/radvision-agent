# CLAUDE.md — RadVision Pro Support Agent

## Project Overview
Agentic RAG chatbot prototype for a fictional medical imaging visualization platform ("RadVision Pro").
Built as a take-home assignment for a Medior AI Engineer role.

The agent supports two personas (Sales Engineers, Support Engineers) in
troubleshooting, configuration, and product questions about a radiology 2D/3D visualization platform.

## Architecture (LangGraph)

### Main Workflow — 5 nodes:
1. **Triage + Route** — Single LLM call: extract entities (version, OS, GPU, error codes, persona),
   classify intent, and decide which source to check first. Three conditional output edges:
   - "error_pattern" → Log Analyzer tool
   - "docs_kb" → RAG subgraph
   - "config_check" → Compatibility Checker tool
2. **Sufficiency Check** — Evaluates gathered evidence. Enough to answer? If not, loops back to
   Triage + Route with refined query (max 1 retry). Conditional: sufficient → continue, insufficient → retry.
3. **Response Synthesizer** — Formats the answer based on persona. Same evidence, different output
   for sales vs support.
4. **Grounding Checker** — Verifies every factual claim in the drafted response against the actual
   retrieved chunks. If ungrounded claims found → send back to synthesizer for regeneration.
5. **Escalation Gate** — Three-way conditional: resolve (output answer), clarify (ask user for more info),
   or escalate (package evidence for human expert). Decision based on grounding confidence + evidence quality.

### RAG Subgraph (separate, does not count toward 5 nodes):
- Query Rewriter → Index Selector (conditional: which collections) → Multi-Index Retrieve → Cross-Index Reranker → Relevance Gate (conditional: pass or retry rewrite, max 1)
- Three vector store collections, each with a different chunking strategy:
  - **KB articles** (Knowledge Base): official, curated troubleshooting documents. Each article covers one known issue with Symptom / Root Cause / Workaround / Resolution. **Chunked by H2 section** — one chunk per section. Each chunk includes the article title and metadata (Article ID, Affected Version, Component, Fixed In) as a preamble. 6 articles × ~5 sections = ~33 chunks. The synthesizer fetches Workaround/Resolution sections directly by metadata filter (not embedding search) to avoid duplication.
  - **Product docs**: component reference documentation with configuration settings. **Chunked by H3 subsection** — one chunk per parameter or scenario. Each chunk includes the component name + full Overview section as a preamble. 4 docs = ~38 chunks.
  - **Release notes**: per-version changelogs. **Chunked by H3 subsection** — one chunk per feature or known-issue entry. 4 versions = ~27 chunks.

### Tools (2, non-retrieval):
1. **Log Pattern Analyzer** — Regex/rule-based matching of error messages against known error signatures.
   Input: error string or log snippet. Output: matched KB id, confidence, pattern name, description.
   Implementation: JSON database of ~30 error patterns with regex matchers.
2. **Compatibility Checker** — Structured lookup: version × OS × GPU × feature → support status.
   Input: query dict. Output: status (supported/limited/unsupported/experimental), reason, recommendation.
   Implementation: JSON matrix. GPU name normalization via `_resolve_gpu()` (e.g., "T4" → "NVIDIA T4").

## Tech Stack
- Python 3.11+
- LangGraph (langgraph) for workflow orchestration
- LangChain for LLM integration and document processing
- ChromaDB for vector storage
- Ollama (gemma3:4b for all LLM nodes — configurable via env vars in src/config.py)
- Sentence-transformers (all-MiniLM-L6-v2) for embeddings
- Streamlit for UI
- Docker + docker-compose for containerization

## Project Structure
```
radvision-agent/
├── CLAUDE.md              # This file
├── README.md              # Full documentation (deliverable)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .gitignore
├── data/
│   ├── seed/              # Product specification YAML (source of truth)
│   │   └── product_spec.yaml
│   ├── corpus/            # Generated documents (from seed)
│   │   ├── kb_articles/
│   │   ├── product_docs/
│   │   └── release_notes/
│   └── vectorstore/       # ChromaDB persistent storage
├── src/
│   ├── __init__.py
│   ├── config.py          # Settings, model config, paths
│   ├── state.py           # LangGraph state definition (TypedDict)
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── graph.py       # Main LangGraph workflow assembly
│   │   ├── triage.py      # Node 1: Triage + Route
│   │   ├── sufficiency.py # Node 2: Sufficiency Check
│   │   ├── synthesizer.py # Node 3: Response Synthesizer
│   │   ├── grounding.py   # Node 4: Grounding Checker
│   │   └── escalation.py  # Node 5: Escalation Gate
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── subgraph.py    # RAG subgraph assembly
│   │   ├── rewriter.py    # Query rewriter node
│   │   ├── retriever.py   # Multi-index retriever
│   │   ├── reranker.py    # Cross-index reranker
│   │   └── ingest.py      # Document ingestion pipeline
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── log_analyzer.py       # Tool 1: Log pattern matching
│   │   └── compat_checker.py     # Tool 2: Compatibility matrix
│   └── ui/
│       └── app.py         # Streamlit application
├── scripts/
│   ├── generate_corpus.py  # Generate synthetic docs from seed
│   ├── ingest_corpus.py    # Ingest into ChromaDB
│   └── run_evaluation.py   # Run eval + load test
├── tests/
│   ├── test_triage.py
│   ├── test_tools.py
│   ├── test_rag.py
│   └── test_integration.py
└── evaluation/
    ├── test_questions.yaml  # 15-20 evaluation questions
    └── load_test.py         # 100-200 query load test
```

## Coding Conventions
- Type hints on all function signatures
- Docstrings on all public functions (Google style)
- Use Pydantic models for structured data (tool inputs/outputs, state)
- Keep each node in its own file, under 150 lines
- Use logging, not print statements
- Environment variables for all configuration (model name, paths, thresholds)

## State Schema (LangGraph)
The shared state flows through all nodes. Key fields:
```python
class AgentState(TypedDict):
    query: str                          # Original user query
    persona: str                        # "sales" | "support"
    entities: dict                      # Extracted: version, os, gpu, error_code, etc.
    intent: str                         # "troubleshooting" | "feature_inquiry" | "config_help"
    route_decision: str                 # "error_pattern" | "docs_kb" | "config_check"
    tool_results: list[dict]            # Outputs from tools
    rag_results: list[dict]             # Retrieved + reranked chunks
    evidence_sufficient: bool           # Sufficiency check result
    retry_count: int                    # Track retries (max 1)
    draft_response: str                 # Synthesizer output
    grounding_score: float              # 0.0-1.0 from grounding checker
    ungrounded_claims: list[str]        # Claims that failed verification
    final_response: str                 # Final output
    outcome: str                        # "resolved" | "clarify" | "escalate"
    escalation_package: dict | None     # Packaged context for human expert
```

## Build Order (phases for Claude Code)
1. State + Config + Project skeleton
2. Tools (log analyzer + compat checker) — these are pure Python, no LLM needed
3. RAG subgraph (ingest pipeline + retrieval + reranking)
4. Main graph nodes (triage, sufficiency, synthesizer, grounding, escalation)
5. Graph assembly + wiring conditional edges
6. Streamlit UI
7. Synthetic data generation from seed
8. Evaluation + load testing
9. Docker + docker-compose
10. README documentation

## Important Constraints
- NO paid APIs. Use Ollama with open-source models only.
- All documents generated from data/seed/product_spec.yaml for internal consistency.
- Streamlit UI should show: chat interface, which nodes executed, retrieved sources, grounding score.
