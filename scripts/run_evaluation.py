"""Evaluation script for the RadVision Pro Support Agent.

Runs all questions in evaluation/test_questions.yaml through the compiled
agent, scores each answer on four dimensions, logs everything to MLflow
(experiment: radvision/evaluation), and saves a JSON report.

Usage (from project root):
    python scripts/run_evaluation.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import yaml

from src.agents.graph import _get_graph
from src.config import LLM_MODEL, RAG_RELEVANCE_THRESHOLD, RAG_TOP_K
from src.tracking import (
    EXPERIMENT_EVALUATION,
    log_artifact_safe,
    log_params_safe,
    set_experiment,
)

EVAL_DIR = Path(__file__).parent.parent / "evaluation"
QUESTIONS_PATH = EVAL_DIR / "test_questions.yaml"
RESULTS_PATH = EVAL_DIR / "results.json"

# Map YAML persona names → graph persona names
_PERSONA_MAP = {
    "support_engineer": "support",
    "sales":            "sales",
    "support":          "support",
}


# ---------------------------------------------------------------------------
# Per-question runner
# ---------------------------------------------------------------------------


def _run_question(graph, q: dict) -> dict:
    """Run one question through the graph; return a scored result dict."""
    persona = _PERSONA_MAP.get(q.get("persona", "support"), "support")
    node_trace: list[str] = []
    state: dict = {
        "query":               q["query"],
        "persona":             persona,
        "retry_count":         0,
        "grounding_regen_count": 0,
    }

    t0 = time.perf_counter()
    for chunk in graph.stream(state, stream_mode="updates"):
        for node_name, updates in chunk.items():
            node_trace.append(node_name)
            state.update(updates)
    latency = round(time.perf_counter() - t0, 3)

    # --- Route correctness ---
    route_correct = int(state.get("route_decision") == q.get("expected_route"))

    # --- Tool correctness ---
    ran_log    = "run_log_analyzer"   in node_trace
    ran_compat = "run_compat_checker" in node_trace
    expected_tools = q.get("expected_tools", [])
    exp_log    = "log_analyzer"    in expected_tools
    exp_compat = "compat_checker"  in expected_tools
    tool_correct = int(ran_log == exp_log and ran_compat == exp_compat)

    # --- Answer quality: fraction of expected keywords in final_response ---
    final = state.get("final_response", "").lower()
    kws = q.get("expected_answer_contains", [])
    answer_quality = (
        round(sum(1 for kw in kws if kw.lower() in final) / len(kws), 3)
        if kws else 1.0
    )

    # --- Outcome correctness ---
    expected_outcome = q.get("expected_outcome")
    actual_outcome   = state.get("outcome", "")
    outcome_correct  = int(actual_outcome == expected_outcome) if expected_outcome else 1

    return {
        "id":                q["id"],
        "query":             q["query"][:80],
        "persona":           persona,
        "difficulty":        q.get("difficulty", ""),
        "expected_route":    q.get("expected_route"),
        "actual_route":      state.get("route_decision"),
        "route_correct":     route_correct,
        "expected_tools":    expected_tools,
        "ran_log_analyzer":  ran_log,
        "ran_compat_checker": ran_compat,
        "tool_correct":      tool_correct,
        "answer_quality":    answer_quality,
        "expected_outcome":  expected_outcome,
        "actual_outcome":    actual_outcome,
        "outcome_correct":   outcome_correct,
        "grounding_score":   round(state.get("grounding_score", 0.0), 3),
        "grounding_pass":    state.get("grounding_pass", False),
        "node_trace":        " → ".join(node_trace),
        "latency_s":         latency,
        "expected_answer":   q.get("expected_answer", ""),
        "actual_answer":     state.get("final_response", ""),
    }


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _print_table(results: list[dict]) -> None:
    header = f"{'ID':<10} {'R✓':>3} {'T✓':>3} {'Qual':>6} {'O✓':>3} {'GScore':>7} {'Lat':>6}  Trace"
    print(header)
    print("─" * 110)
    for r in results:
        route_icon   = "✓" if r["route_correct"]   else "✗"
        tool_icon    = "✓" if r["tool_correct"]    else "✗"
        outcome_icon = "✓" if r["outcome_correct"] else ("–" if r["expected_outcome"] is None else "✗")
        trace_short  = r["node_trace"].replace("run_log_analyzer", "log_A")  \
                                      .replace("run_compat_checker", "compat")  \
                                      .replace("run_rag", "rag")
        print(
            f"{r['id']:<10} {route_icon:>3} {tool_icon:>3} "
            f"{r['answer_quality']:>6.3f} {outcome_icon:>3} "
            f"{r['grounding_score']:>7.3f} {r['latency_s']:>5.2f}s  "
            f"{trace_short[:60]}"
        )


def _aggregate(results: list[dict]) -> dict:
    n = len(results)
    return {
        "num_questions":    n,
        "route_accuracy":   round(sum(r["route_correct"]   for r in results) / n, 3),
        "tool_accuracy":    round(sum(r["tool_correct"]    for r in results) / n, 3),
        "answer_quality_mean": round(sum(r["answer_quality"] for r in results) / n, 3),
        "outcome_accuracy": round(sum(r["outcome_correct"] for r in results) / n, 3),
        "avg_latency_s":    round(sum(r["latency_s"]       for r in results) / n, 3),
        "total_latency_s":  round(sum(r["latency_s"]       for r in results), 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    questions = yaml.safe_load(QUESTIONS_PATH.read_text())["questions"]
    graph = _get_graph()

    print(f"\nRadVision Pro — Evaluation ({len(questions)} questions)\n")

    set_experiment(EXPERIMENT_EVALUATION)
    with mlflow.start_run(run_name="eval-batch") as run:
        log_params_safe({
            "num_questions":    len(questions),
            "llm_model":        LLM_MODEL,
            "rag_threshold":    RAG_RELEVANCE_THRESHOLD,
            "top_k":            RAG_TOP_K,
        })

        results = []
        for i, q in enumerate(questions):
            print(f"  [{i+1:02d}/{len(questions)}] {q['id']} — {q['query'][:60]}")
            r = _run_question(graph, q)
            results.append(r)

            # Per-question metrics with step index
            mlflow.log_metrics({
                "route_correct":   r["route_correct"],
                "tool_correct":    r["tool_correct"],
                "answer_quality":  r["answer_quality"],
                "outcome_correct": r["outcome_correct"],
                "grounding_score": r["grounding_score"],
                "latency_s":       r["latency_s"],
            }, step=i)

        agg = _aggregate(results)
        mlflow.log_metrics({
            "route_accuracy":       agg["route_accuracy"],
            "tool_accuracy":        agg["tool_accuracy"],
            "answer_quality_mean":  agg["answer_quality_mean"],
            "outcome_accuracy":     agg["outcome_accuracy"],
            "avg_latency_s":        agg["avg_latency_s"],
        })

        # Save and log JSON artifact
        RESULTS_PATH.write_text(json.dumps(
            {"run_id": run.info.run_id, "aggregate": agg, "questions": results},
            indent=2,
        ))
        log_artifact_safe(str(RESULTS_PATH))

    print(f"\n{'─' * 110}")
    _print_table(results)
    print(f"\n{'─' * 110}")
    print(
        f"AGGREGATE  route={agg['route_accuracy']:.0%}  "
        f"tool={agg['tool_accuracy']:.0%}  "
        f"quality={agg['answer_quality_mean']:.3f}  "
        f"outcome={agg['outcome_accuracy']:.0%}  "
        f"avg_lat={agg['avg_latency_s']:.2f}s"
    )
    print(f"\nMLflow run : {run.info.run_id}")
    print(f"Results    : {RESULTS_PATH}")


if __name__ == "__main__":
    main()
