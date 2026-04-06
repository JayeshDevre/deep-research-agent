"""
Tier 2 — Episodic Memory (session summaries)
Tier 3 — Semantic Memory (extracted facts)

Both live in the same ChromaDB collection, distinguished by metadata type:
  type = "session_summary" | "fact"
"""

import uuid
from datetime import datetime, timezone
from typing import Any

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from agent.config import CHROMA_COLLECTION, CHROMA_N_RESULTS
from agent.utils import get_logger

logger = get_logger(__name__)


class EpisodicMemory:
    def __init__(self, persist_path: str = "./chroma_db") -> None:
        try:
            self._client = chromadb.PersistentClient(
                path=persist_path,
                settings=Settings(anonymized_telemetry=False),
            )
            self._ef = embedding_functions.DefaultEmbeddingFunction()
            self._collection = self._client.get_or_create_collection(
                name=CHROMA_COLLECTION,
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("EpisodicMemory ready | path=%s | chunks=%d", persist_path, self.count())
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialise ChromaDB at '{persist_path}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def store_session_summary(self, summary: str, session_id: str | None = None) -> None:
        """Persist a compressed session summary for future retrieval."""
        self._upsert(
            doc_id=session_id or str(uuid.uuid4()),
            document=summary,
            metadata={"type": "session_summary"},
        )

    def store_facts_bulk(self, facts: list[str], source: str = "extracted") -> None:
        """Store multiple extracted facts (Tier 3 semantic memory)."""
        if not facts:
            return
        ids   = [str(uuid.uuid4()) for _ in facts]
        metas = [
            {"type": "fact", "source": source, "timestamp": _now()}
            for _ in facts
        ]
        try:
            self._collection.upsert(ids=ids, documents=facts, metadatas=metas)
            logger.debug("Stored %d facts", len(facts))
        except Exception:
            logger.exception("Failed to store facts")

    def _upsert(self, doc_id: str, document: str, metadata: dict[str, Any]) -> None:
        full_meta = {**metadata, "timestamp": _now()}
        try:
            self._collection.upsert(
                ids=[doc_id],
                documents=[document],
                metadatas=[full_meta],
            )
            logger.debug("Upserted doc_id=%s type=%s", doc_id, metadata.get("type"))
        except Exception:
            logger.exception("Failed to upsert document id=%s", doc_id)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(self, text: str, n: int = CHROMA_N_RESULTS) -> list[dict[str, Any]]:
        """
        Semantic search over stored summaries and facts.

        Returns:
            List of {"text", "source", "score", "type"} sorted by relevance.
        """
        total = self.count()
        if total == 0:
            return []

        try:
            results = self._collection.query(
                query_texts=[text],
                n_results=min(n, total),
            )
        except Exception:
            logger.exception("ChromaDB query failed for text='%s'", text[:60])
            return []

        chunks: list[dict[str, Any]] = []
        documents  = results.get("documents",  [[]])[0]
        metadatas  = results.get("metadatas",  [[]])[0]
        distances  = results.get("distances",  [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            chunks.append({
                "text":   doc,
                "source": f"memory:{meta.get('type', 'unknown')}",
                "score":  round(1.0 - dist, 4),
                "type":   meta.get("type"),
            })

        logger.debug("Memory query returned %d chunks", len(chunks))
        return chunks

    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception:
            return 0

    def count_facts(self) -> int:
        try:
            result = self._collection.get(where={"type": "fact"}, include=[])
            return len(result["ids"])
        except Exception:
            return 0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
