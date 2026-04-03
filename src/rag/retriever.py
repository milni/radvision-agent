"""RAG retriever for the RadVision Pro support agent.

Queries one or more ChromaDB collections and returns merged, score-sorted results.
Uses ChromaDB's built-in DefaultEmbeddingFunction (all-MiniLM-L6-v2 via ONNX),
the same model used during ingestion, so no separate embedder is needed.
"""

import logging
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from src.config import RAG_TOP_K, VECTORSTORE_DIR

logger = logging.getLogger(__name__)


class Retriever:
    """Queries ChromaDB collections and returns ranked result chunks.

    Supports querying a single collection or multiple collections at once.
    Results from multiple collections are merged and re-ranked by score.
    """

    def __init__(self, vectorstore_dir: Path = VECTORSTORE_DIR) -> None:
        self._client = chromadb.PersistentClient(path=str(vectorstore_dir))
        self._ef = DefaultEmbeddingFunction()
        logger.info("Retriever ready")

    def retrieve(
        self,
        query: str,
        collections: list[str],
        top_k: int = RAG_TOP_K,
    ) -> list[dict]:
        """Query one or more collections and return the top-k results by score.

        Args:
            query: The search query text.
            collections: Collection names to search (e.g. ["kb_articles", "past_tickets"]).
            top_k: Maximum number of results to return across all collections.

        Returns:
            List of result dicts sorted by score descending, each containing:
                text              — the chunk text
                metadata          — ChromaDB metadata dict (source_type, kb_id, etc.)
                score             — cosine similarity (0.0–1.0, higher is better)
                source_collection — which collection the chunk came from
        """
        results: list[dict] = []
        for name in collections:
            results.extend(self._query_collection(name, query, top_k))

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def _query_collection(self, collection_name: str, query: str, top_k: int) -> list[dict]:
        """Query a single collection. Returns an empty list on any error."""
        try:
            collection = self._client.get_collection(
                name=collection_name,
                embedding_function=self._ef,
            )
        except Exception:
            logger.warning("Collection '%s' not found — skipping.", collection_name)
            return []

        count = collection.count()
        if count == 0:
            logger.warning("Collection '%s' is empty — skipping.", collection_name)
            return []

        response = collection.query(
            query_texts=[query],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        results = []
        for doc, meta, dist in zip(
            response["documents"][0],
            response["metadatas"][0],
            response["distances"][0],
        ):
            # ChromaDB cosine distance = 1 - cosine_similarity
            results.append({
                "text": doc,
                "metadata": meta,
                "score": round(1.0 - dist, 4),
                "source_collection": collection_name,
            })

        return results
