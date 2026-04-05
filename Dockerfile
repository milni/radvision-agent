FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies — copied first for layer caching
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Application code
COPY . .

# Generate corpus from seed YAML
RUN python scripts/generate_corpus.py

# Ingest corpus into ChromaDB (embeddings run locally — no Ollama needed at build time)
RUN python scripts/ingest_corpus.py

EXPOSE 8501

# Use Python-based healthcheck (curl is not present in python:3.11-slim)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
