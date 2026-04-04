"""Async load test for the RadVision Pro Support Agent.

Sends a configurable number of concurrent queries and reports p50/p95/p99
latency, queries-per-second, and error rate.  Results are logged to MLflow
(experiment: radvision/evaluation).

Usage (from project root):
    python evaluation/load_test.py                  # default: 50 queries, 10 concurrent
    python evaluation/load_test.py --queries 100 --concurrency 20
"""

import argparse
import asyncio
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow

from src.agents.graph import _get_graph
from src.config import LLM_MODEL
from src.tracking import EXPERIMENT_EVALUATION, log_metrics_safe, log_params_safe, set_experiment

# ---------------------------------------------------------------------------
# Query pool — diverse enough to exercise different code paths
# ---------------------------------------------------------------------------

_QUERY_POOL = [
    ("DICOM association rejected with TLS renegotiation error on v4.2", "support"),
    ("cardiac 4D rendering visual artifacts after upgrading to v4.2.1", "support"),
    ("FHIR endpoint returns 503 during peak load on v4.2", "support"),
    ("what new features were introduced in version 4.2", "sales"),
    ("Is NVIDIA T4 supported for vessel analysis in v4.2", "sales"),
    ("how to configure SCP_TLS_RENEGOTIATION in the DICOM gateway", "support"),
    ("VRAM_FALLBACK_TRIGGERED in render logs, cardiac images look wrong", "support"),
    ("screen sharing lag in collaboration hub on RHEL 9", "support"),
    ("can RadVision Pro integrate with our Epic EHR system", "sales"),
    ("AssocReject reason=0x0006 in DICOM gateway logs after F5 upgrade", "support"),
]


def _run_single(graph, query: str, persona: str) -> tuple[float, bool]:
    """Run one query; return (latency_seconds, success)."""
    state = {
        "query":               query,
        "persona":             persona,
        "retry_count":         0,
        "grounding_regen_count": 0,
    }
    t0 = time.perf_counter()
    try:
        for chunk in graph.stream(state, stream_mode="updates"):
            for _, updates in chunk.items():
                state.update(updates)
        return time.perf_counter() - t0, True
    except Exception as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)
        return time.perf_counter() - t0, False


async def _run_load_test(
    num_queries: int,
    concurrency: int,
) -> tuple[list[float], int]:
    """Run num_queries concurrently (up to concurrency at a time).

    Returns (latencies, error_count).
    """
    graph = _get_graph()
    pool = _QUERY_POOL * (num_queries // len(_QUERY_POOL) + 1)
    work = pool[:num_queries]

    loop = asyncio.get_event_loop()
    latencies: list[float] = []
    errors = 0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            loop.run_in_executor(executor, _run_single, graph, q, p)
            for q, p in work
        ]
        for fut in asyncio.as_completed(futures):
            lat, ok = await fut
            latencies.append(lat)
            if not ok:
                errors += 1
            done = len(latencies)
            if done % 10 == 0:
                print(f"  {done}/{num_queries} completed …")

    return latencies, errors


def _percentile(data: list[float], p: float) -> float:
    """Return the p-th percentile (0–100) of data."""
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)


def main(num_queries: int = 50, concurrency: int = 10) -> None:
    print(f"\nRadVision Pro — Load Test")
    print(f"  queries={num_queries}  concurrency={concurrency}  model={LLM_MODEL}\n")

    wall_start = time.perf_counter()
    latencies, errors = asyncio.run(_run_load_test(num_queries, concurrency))
    wall_elapsed = time.perf_counter() - wall_start

    n = len(latencies)
    p50  = _percentile(latencies, 50)
    p95  = _percentile(latencies, 95)
    p99  = _percentile(latencies, 99)
    mean = statistics.mean(latencies)
    qps  = round(n / wall_elapsed, 2)
    err_rate = round(errors / n, 3) if n else 0.0

    # MLflow
    set_experiment(EXPERIMENT_EVALUATION)
    with mlflow.start_run(run_name="load-test") as run:
        log_params_safe({
            "num_queries":  num_queries,
            "concurrency":  concurrency,
            "llm_model":    LLM_MODEL,
        })
        log_metrics_safe({
            "latency_p50_ms":      round(p50  * 1000, 1),
            "latency_p95_ms":      round(p95  * 1000, 1),
            "latency_p99_ms":      round(p99  * 1000, 1),
            "latency_mean_ms":     round(mean * 1000, 1),
            "queries_per_second":  qps,
            "error_rate":          err_rate,
            "total_queries":       n,
        })

    # Report
    print(f"\n{'─' * 55}")
    print(f"  Queries completed : {n}  (errors: {errors})")
    print(f"  Wall time         : {wall_elapsed:.1f}s")
    print(f"  Throughput        : {qps} q/s")
    print(f"  Error rate        : {err_rate:.1%}")
    print(f"  Latency p50       : {p50*1000:.0f} ms")
    print(f"  Latency p95       : {p95*1000:.0f} ms")
    print(f"  Latency p99       : {p99*1000:.0f} ms")
    print(f"  Latency mean      : {mean*1000:.0f} ms")
    print(f"{'─' * 55}")
    print(f"\nMLflow run: {run.info.run_id}")

    # Bottleneck hint (dummy LLM → embedding is the likely bottleneck)
    if p95 > 2.0:
        print("\nBottleneck hint: p95 > 2s — likely ChromaDB embedding lookup.")
        print("  Optimization 1: increase RAG_TOP_K batch size to amortise ONNX overhead.")
        print("  Optimization 2: cache the Retriever instance (already singleton in graph.py).")
    else:
        print("\nLatency looks healthy for a dummy-LLM configuration.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load test the RadVision Pro agent")
    parser.add_argument("--queries",     type=int, default=50,  help="Total queries to send")
    parser.add_argument("--concurrency", type=int, default=10,  help="Max concurrent workers")
    args = parser.parse_args()
    main(num_queries=args.queries, concurrency=args.concurrency)
