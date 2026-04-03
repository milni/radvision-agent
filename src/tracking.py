"""MLflow tracking utilities for the RadVision Pro support agent.

Three experiments map to the three layers of the system:

  radvision/ingestion   — corpus ingestion runs (chunk counts, timing)
  radvision/rag         — per-query RAG subgraph runs (scores, collections)
  radvision/agent       — full agent query runs (route, grounding, outcome)
  radvision/evaluation  — evaluation batch runs (accuracy, latency)

All public functions are safe to call when no MLflow run is active —
they silently no-op so production code never breaks if tracking is
disabled or the tracking server is unreachable.

Typical usage
-------------
Ingestion script (explicit run):
    with mlflow.start_run(experiment_id=set_experiment(EXPERIMENT_INGESTION)):
        log_params({"embedding_model": ..., "corpus_dir": ...})
        log_metrics({"chunks.kb_articles": 6, ...})

Agent run (wraps graph invocation):
    with mlflow.start_run(experiment_id=set_experiment(EXPERIMENT_AGENT)):
        result = graph.invoke(state)          # nodes log into the active run
        log_metrics({"outcome": ..., ...})

RAG subgraph / node (no-op when no run is active):
    log_metrics_safe({"rag.top_score": 0.82, "rag.rewrite_count": 0})
"""

import logging

import mlflow

logger = logging.getLogger(__name__)

# Experiment names — one per system layer
EXPERIMENT_INGESTION = "radvision/ingestion"
EXPERIMENT_RAG = "radvision/rag"
EXPERIMENT_AGENT = "radvision/agent"
EXPERIMENT_EVALUATION = "radvision/evaluation"


def set_experiment(name: str) -> str:
    """Create-or-get an MLflow experiment and return its ID."""
    mlflow.set_experiment(name)
    return mlflow.get_experiment_by_name(name).experiment_id


def log_params_safe(params: dict) -> None:
    """Log params to the active run. No-op if no run is active."""
    try:
        if mlflow.active_run():
            mlflow.log_params(params)
    except Exception as exc:
        logger.debug("MLflow log_params skipped: %s", exc)


def log_metrics_safe(metrics: dict) -> None:
    """Log metrics to the active run. No-op if no run is active."""
    try:
        if mlflow.active_run():
            mlflow.log_metrics(metrics)
    except Exception as exc:
        logger.debug("MLflow log_metrics skipped: %s", exc)


def log_artifact_safe(local_path: str) -> None:
    """Log a file artifact to the active run. No-op if no run is active."""
    try:
        if mlflow.active_run():
            mlflow.log_artifact(local_path)
    except Exception as exc:
        logger.debug("MLflow log_artifact skipped: %s", exc)


def set_tags_safe(tags: dict) -> None:
    """Set tags on the active run. No-op if no run is active."""
    try:
        if mlflow.active_run():
            mlflow.set_tags(tags)
    except Exception as exc:
        logger.debug("MLflow set_tags skipped: %s", exc)
