"""Configuration for the RadVision Pro support agent.

All settings are configurable via environment variables with sensible defaults.
This allows the same codebase to run locally (with dummy LLM) or in Docker (with Ollama).
"""

import os
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SEED_PATH = DATA_DIR / "seed" / "product_spec.yaml"
CORPUS_DIR = DATA_DIR / "corpus"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

# --- LLM ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral:7b-instruct-v0.3-q4_K_M")
USE_DUMMY_LLM = os.getenv("USE_DUMMY_LLM", "true").lower() == "true"
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# --- Embeddings ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# --- RAG ---
CHROMA_COLLECTION_NAMES = ["kb_articles", "product_docs", "release_notes", "past_tickets"]
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_RELEVANCE_THRESHOLD = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.3"))
RAG_MAX_RETRIES = int(os.getenv("RAG_MAX_RETRIES", "1"))

# --- Agent ---
SUFFICIENCY_MAX_RETRIES = int(os.getenv("SUFFICIENCY_MAX_RETRIES", "1"))
GROUNDING_THRESHOLD = float(os.getenv("GROUNDING_THRESHOLD", "0.6"))
GROUNDING_MAX_REGENERATIONS = int(os.getenv("GROUNDING_MAX_REGENERATIONS", "1"))
