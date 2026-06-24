"""
SiliconMind Retrieval — v2.0
─────────────────────────────
Sprint 2: Confidence scoring stolen from ORAssistant.
Every answer now signals whether it came from your documents or general knowledge.
"""

import os


def init_vectorstore(path: str = "./vectorstore"):
    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False)
        )
        return client.get_or_create_collection(
            name="siliconmind_docs",
            metadata={"hnsw:space": "cosine"}
        )
    except ImportError:
        return None


def retrieve(question: str, vector_store, n_results: int = 5,
             topic_filter: str = None) -> tuple[str, list[str], dict]:
    """
    Semantic search over ChromaDB.
    Returns (context_text, sources, confidence_info)

    confidence_info = {
        "level":       "high" | "medium" | "low" | "none",
        "label":       human-readable string for the UI,
        "score":       float 0–1 (best similarity score found),
        "from_docs":   bool
    }
    """
    if not vector_store:
        return "", [], {"level": "none", "label": "No documents indexed", "score": 0.0, "from_docs": False}

    try:
        where = {"topics": {"$contains": topic_filter}} if topic_filter else None
        kwargs = {
            "query_texts": [question],
            "n_results":   n_results,
            "include":     ["documents", "metadatas", "distances"]   # distances = 1 - cosine_similarity
        }
        if where:
            kwargs["where"] = where

        results   = vector_store.query(**kwargs)
        docs      = results.get("documents",  [[]])[0]
        metas     = results.get("metadatas",  [[]])[0]
        distances = results.get("distances",  [[]])[0]   # 0 = perfect match, 1 = no similarity

        if not docs:
            return "", [], {"level": "none", "label": "No documents indexed", "score": 0.0, "from_docs": False}

        # Convert distance → similarity score (cosine space: similarity = 1 - distance)
        similarities = [round(1.0 - d, 3) for d in distances]
        best_score   = max(similarities) if similarities else 0.0

        # ── Confidence thresholds (from ORAssistant paper + tuned for DFT) ──
        if best_score >= 0.75:
            level = "high"
            label = f"High confidence — answer grounded in your documents (similarity {best_score:.2f})"
        elif best_score >= 0.50:
            level = "medium"
            label = f"Medium confidence — partial match in your documents (similarity {best_score:.2f})"
        elif best_score >= 0.30:
            level = "low"
            label = f"Low confidence — weak document match; answer may use general knowledge (similarity {best_score:.2f})"
        else:
            level = "none"
            label = "Answering from general knowledge — no relevant match found in your documents"

        confidence = {
            "level":     level,
            "label":     label,
            "score":     best_score,
            "from_docs": best_score >= 0.50
        }

        sources = list({m.get("source", "Unknown") for m in metas})

        # Format context with source labels and similarity scores
        context_parts = []
        for doc, meta, sim in zip(docs, metas, similarities):
            src    = meta.get("source", "Unknown")
            topics = meta.get("topics", "")
            context_parts.append(
                f"[Source: {src} | Topic: {topics} | Similarity: {sim}]\n{doc}"
            )

        return "\n\n---\n\n".join(context_parts), sources, confidence

    except Exception as e:
        return "", [], {"level": "none", "label": f"Retrieval error: {e}", "score": 0.0, "from_docs": False}
