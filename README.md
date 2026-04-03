# RadVision Pro — Agentic RAG Support Agent

An agentic RAG chatbot prototype built with LangGraph that assists Sales Engineers, Support Engineers, and Technical Field Engineers working with a medical imaging visualization platform.

## Problem Statement

### Why this domain?

Medical imaging visualization platforms generate complex multi-dimensional support queries. A single question like "3D rendering shows artifacts on cardiac CTs" could stem from a GPU hardware limitation, a software configuration issue, a known bug in a specific version, or a DICOM integration problem. Simple RAG cannot handle this — the system needs to reason about which information source to check first, whether the gathered evidence is sufficient, and whether to escalate when confidence is low.

### Why agentic RAG?

Three reasons a flat retrieval pipeline falls short:

1. **The first retrieval is often not enough.** A troubleshooting query may initially match a KB article, but the real root cause turns out to be a hardware compatibility issue discoverable only through a structured tool lookup. The agent must assess sufficiency and retry with a different strategy.

2. **Different query types need fundamentally different pipelines.** An error code in a log file should go to a pattern matcher (not vector search). A "does GPU X support feature Y?" question should go to a compatibility matrix (not documentation retrieval). A "how do I configure DICOM?" question should go to product docs. The routing decision is non-trivial.

3. **The same evidence needs different framing per persona.** A field engineer needs exact config file paths and commands. A sales engineer needs customer-facing positioning language. This is more than a prompt template — the level of technical detail, the inclusion of workarounds vs. upgrade recommendations, and the escalation threshold all differ.

## Architecture

### Main Workflow — 5 LangGraph Nodes

| Node | Name | Type | Purpose |
|------|------|------|---------|
| 1 | Triage + Route | Conditional | Parse entities, classify intent, route to error_pattern / docs_kb / config_check |
| 2 | Sufficiency Check | Conditional | Evaluate gathered evidence. Sufficient → continue. Insufficient → retry via different route (max 1) |
| 3 | Response Synthesizer | Processing | Generate persona-shaped answer from evidence |
| 4 | Grounding Checker | Conditional | Verify every claim against retrieved sources. Ungrounded → regenerate |
| 5 | Escalation Gate | Conditional | Three-way output: resolve / ask for clarification / escalate to human |

### RAG Subgraph (separate)

Query Rewriter → Index Selector → Multi-Index Retrieve (4 collections) → Cross-Index Reranker → Relevance Gate

Four vector store collections with different chunking strategies:
- **KB Articles**: chunked by solution section
- **Product Docs**: chunked by paragraph with heading context
- **Release Notes**: chunked by version entry
- **Past Tickets**: chunked by resolution description

### Tools (non-retrieval)

1. **Log Pattern Analyzer**: regex/rule-based matching of error messages against known error signatures. Returns matched KB reference, confidence, and structured diagnosis.
2. **Compatibility Checker**: structured lookup against version × OS × GPU × feature matrix. Returns support status, reason, and recommendation.

### Architecture Diagram

<!-- TODO: Add mermaid diagram -->

## Design Decisions

### Model Choice

<!-- TODO: Document model selection trade-offs -->

### Chunking Strategy

<!-- TODO: Document why different collections use different chunking -->

### Synthetic Data

All documents are generated from a single product specification seed (`data/seed/product_spec.yaml`). This ensures referential integrity: KB article IDs referenced in release notes match actual KB articles, compatibility matrix entries align with known issues, and past ticket resolutions cite real configuration parameters.

## Evaluation

### Functional Evaluation

15-question test set covering:
- All three routing paths (error_pattern, docs_kb, config_check)
- Multi-step queries requiring the sufficiency retry loop
- Persona differentiation (same query, different persona)
- Edge cases: vague queries (→ clarify), security issues (→ escalate), nonexistent features (→ honest "not available")
- Grounding checker validation (detecting hallucinated KB references)

### Results

<!-- TODO: Fill in after running evaluation -->

### Load Test

<!-- TODO: 100-200 queries, latency percentiles, bottleneck analysis -->

## Setup & Running

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with drivers (for Ollama) OR set `USE_DUMMY_LLM=true`

### Quick Start

```bash
# With GPU (full LLM)
docker-compose up --build

# Without GPU (dummy LLM for testing)
docker build -t radvision-agent .
docker run -p 8501:8501 -e USE_DUMMY_LLM=true radvision-agent
```

Open http://localhost:8501

### Local Development

```bash
pip install -e ".[dev]"
python scripts/generate_corpus.py
python scripts/ingest_corpus.py
streamlit run src/ui/app.py
```

### Running Tests

```bash
pytest tests/
python scripts/run_evaluation.py
```
