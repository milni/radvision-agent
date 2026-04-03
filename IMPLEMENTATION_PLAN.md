# Implementation Plan — Claude Code Sessions

Use this as your roadmap. Each phase is one Claude Code session.
Complete each phase fully before moving to the next.

## Phase 1: Tools (no LLM needed)
**Goal:** Both tools working with tests.

```
Build the log_analyzer tool in src/tools/log_analyzer.py:
- Load error patterns from data/seed/product_spec.yaml
- Function: analyze_log(error_text: str) -> dict with keys:
  matched (bool), pattern_id, kb_ref, component, severity, description
- Handle no-match case gracefully
- Write tests in tests/test_tools.py

Then build compat_checker in src/tools/compat_checker.py:
- Load compatibility matrix from product_spec.yaml
- Function: check_compatibility(version, gpu, feature) -> dict with keys:
  status (supported/limited/unsupported), reason, recommendation
- Also: check_platform(version, os) -> dict
- Write tests
```

## Phase 2: Synthetic Data Generation
**Goal:** Full corpus generated from seed.

```
Build scripts/generate_corpus.py:
- Read data/seed/product_spec.yaml
- Generate into data/corpus/:
  - kb_articles/: one markdown file per known_issue (KB-XXXX.md)
    Format: Title, Affected Versions, Symptom, Root Cause, Workaround, Resolution
  - product_docs/: markdown files for each component
    (dicom_gateway.md, rendering_engine.md, integration_gateway.md, collab_hub.md)
    Include: description, configuration reference, all key_settings with defaults
  - release_notes/: one file per version (v4.0.md, v4.1.md, v4.2.md, v4.2.1.md)
  - past_tickets/: one markdown file per ticket template
- Print summary of generated files
```

## Phase 3: RAG Ingestion Pipeline
**Goal:** Documents chunked and stored in ChromaDB.

```
Build src/rag/ingest.py:
- Load documents from data/corpus/
- Different chunking per collection:
  - KB articles: split by section (Symptom / Root Cause / Workaround)
  - Product docs: split by paragraph, prepend heading hierarchy as context
  - Release notes: split by version entry
  - Past tickets: split by resolution section
- Embed with sentence-transformers (all-MiniLM-L6-v2)
- Store in ChromaDB with metadata (source_type, version, component, kb_id)
- Build scripts/ingest_corpus.py that calls this

Then build src/rag/retriever.py:
- Function: retrieve(query, collections, top_k) -> list[dict]
- Each result: {text, metadata, score, source_collection}
- Support querying one or multiple collections
```

## Phase 4: RAG Subgraph
**Goal:** Complete RAG subgraph as a callable LangGraph subgraph.

```
Build the RAG subgraph in src/rag/subgraph.py:
- Query Rewriter node: expand query with domain synonyms
  (e.g., "DICOM timeout" -> "DICOM association timeout SCP connection")
- Index Selector node: based on query type, select which collections
  (troubleshooting -> kb_articles + past_tickets,
   feature question -> product_docs,
   version question -> release_notes,
   unclear -> all four)
- Retriever node: call multi-index retrieve
- Reranker node: merge results from multiple collections, deduplicate, re-score
- Relevance Gate: if top result score < threshold, rewrite and retry (max 1)
- Wire as LangGraph StateGraph, expose as compiled subgraph
```

## Phase 5: Main Graph Nodes
**Goal:** All 5 nodes implemented.

MLflow (experiment: radvision/agent):
- Each node logs its key output as metrics into the active run started
  by run_agent() in Phase 6 (safe no-op if no run is active):
  - triage.py     → tags: intent, route_decision; params: persona
  - sufficiency.py→ metrics: retry_count; tags: evidence_sufficient
  - synthesizer.py→ (no metrics needed, text output)
  - grounding.py  → metrics: grounding_score, ungrounded_claims_count
  - escalation.py → tags: outcome

```
Build each node in src/agents/:

1. triage.py - Triage + Route:
   - LLM call (or dummy) that returns: entities dict, intent, route_decision
   - Prompt: extract version/os/gpu/error_code/component from query, classify intent,
     decide route. Output as JSON.
   - For dummy LLM: use keyword matching (regex for error codes -> error_pattern,
     "compatible"/"support" -> config_check, else -> docs_kb)

2. sufficiency.py - Sufficiency Check:
   - Evaluate: are tool_results + rag_results enough to answer the query?
   - Heuristics for dummy LLM: has_tool_match OR has_rag_above_threshold -> sufficient
   - If insufficient and retry_count < 1: set retry_count += 1, suggest alternative route
   - If insufficient and retry_count >= 1: mark as insufficient, continue anyway

3. synthesizer.py - Response Synthesizer:
   - LLM call that takes evidence + persona and drafts a response
   - Persona templates for dummy LLM:
     - field_engineer: include config paths, exact settings, workaround commands
     - support: step-by-step, include KB references
     - sales: feature-focused, upgrade recommendations, no technical details

4. grounding.py - Grounding Checker:
   - Compare each sentence in draft_response against rag_results + tool_results
   - Score: fraction of claims that have supporting evidence
   - If score < threshold: flag ungrounded claims, send back to synthesizer
   - For dummy LLM: check if KB-XXXX references in response exist in results

5. escalation.py - Escalation Gate:
   - Decision logic:
     - grounding_score > 0.6 AND evidence_sufficient -> "resolved"
     - entities missing key info OR query too vague -> "clarify"
     - security/safety keywords OR grounding_score < 0.3 -> "escalate"
   - Package escalation context: query, evidence gathered, what was tried
```

## Phase 6: Graph Assembly
**Goal:** Complete wired LangGraph with all conditional edges.

MLflow (experiment: radvision/agent):
- run_agent(query, persona) wraps the graph.invoke() call in
  mlflow.start_run(). All nodes (Phase 5) and the RAG subgraph log
  into this parent run automatically via log_metrics_safe().
- Log params: query (truncated to 200 chars), persona, llm_model
- Log final metrics: outcome, grounding_score, total_retry_count

```
Build src/agents/graph.py:
- Create StateGraph with AgentState
- Add all 5 nodes + tool executor nodes + RAG subgraph
- Wire conditional edges:
  - triage -> {error_pattern: log_analyzer, docs_kb: rag_subgraph, config_check: compat_checker}
  - tool/rag results -> sufficiency_check
  - sufficiency_check -> {sufficient: synthesizer, insufficient: triage} (with retry guard)
  - synthesizer -> grounding_checker
  - grounding_checker -> {pass: escalation_gate, fail: synthesizer} (with regen guard)
  - escalation_gate -> {resolved: END, clarify: END, escalate: END}
- Compile graph
- Add function: run_agent(query: str, persona: str) -> dict
- Write integration test in tests/test_integration.py
```

## Phase 7: Streamlit UI
**Goal:** Working chat interface showing agent internals.

MLflow:
- Add a sidebar link to the MLflow UI (mlflow ui → localhost:5000)
- After each query, display the MLflow run ID so the user can drill
  into the full trace in the MLflow UI

```
Build src/ui/app.py:
- Chat interface with message history
- Sidebar: persona selector (Sales / Support / Field Engineer)
- After each query, show expandable sections:
  - "Triage Decision": extracted entities, intent, chosen route
  - "Evidence Gathered": tool results + retrieved chunks with scores
  - "Sufficiency": pass/retry, retry reason if applicable
  - "Grounding": score, any flagged claims
  - "Outcome": resolved/clarify/escalate
- Show the LangGraph execution trace (which nodes ran in what order)
```

## Phase 8: Evaluation & Load Test
**Goal:** All evaluation results documented.

MLflow (experiment: radvision/evaluation):
- One MLflow run per evaluation batch
- params: num_questions, llm_model, rag_threshold, top_k
- metrics per question (step=question_index):
    route_correct, tool_correct, answer_quality, outcome_correct
- aggregate metrics: route_accuracy, answer_quality_mean, outcome_accuracy
- artifacts: evaluation/results.json
- Load test: log latency_p50_ms, latency_p95_ms, latency_p99_ms,
    queries_per_second, error_rate

```
Build scripts/run_evaluation.py:
- Load evaluation/test_questions.yaml
- Run each question through the agent
- Score:
  - Route accuracy: did it pick the expected route?
  - Tool usage: did it call the expected tools?
  - Answer quality: does the response contain expected keywords?
  - Persona correctness: is the tone/depth appropriate?
  - Outcome correctness: resolve/clarify/escalate as expected?
- Print results table and save to evaluation/results.json

Build evaluation/load_test.py:
- Use asyncio (or locust) to send 100-200 queries concurrently
- Measure: p50, p95, p99 latency
- Identify bottleneck (likely LLM inference or embedding)
- Document 1-2 optimization suggestions
```

## Phase 9: Polish & Documentation
**Goal:** Everything clean, documented, reproducible.

```
- Fill in all TODO sections in README.md
- Add mermaid architecture diagram to README
- Ensure docker-compose up works end-to-end
- Run ruff for code formatting
- Run full test suite
- Final git commit with clean history
```
