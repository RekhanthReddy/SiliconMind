"""
SiliconMind Pinecone Store
──────────────────────────
Persistent vector storage using Pinecone.
Documents survive app restarts and cloud redeployments.
"""

import os
from pinecone import Pinecone


def get_index():
    """Connect to Pinecone index."""
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    return pc.Index(host=os.environ.get("PINECONE_HOST"))


def upsert_chunks(chunks: list[dict]) -> int:
    """
    Upload document chunks to Pinecone.
    Each chunk: {"text": str, "source": str, "topics": str, "chunk_idx": int}
    Uses Pinecone's integrated embedding (llama-text-embed-v2) — no separate embedding model needed.
    """
    if not chunks:
        return 0

    index = get_index()

    records = []
    for chunk in chunks:
        record_id = f"{chunk['source'].replace(' ','_')}__chunk_{chunk['chunk_idx']}"
        records.append({
            "_id":    record_id,
            "text":   chunk["text"],
            "source": chunk["source"],
            "topics": chunk.get("topics", "General")
        })

    # Upsert in batches of 50
    batch_size = 50
    for i in range(0, len(records), batch_size):
        index.upsert_records(
            namespace="siliconmind_docs",
            records=records[i:i + batch_size]
        )

    return len(records)


def search_chunks(query: str, n_results: int = 5) -> tuple[str, list[str], dict]:
    """
    Search Pinecone for relevant chunks.
    Returns (context_text, sources, confidence_info)
    """
    if not query:
        return "", [], {"level": "none", "label": "No query provided", "score": 0.0, "from_docs": False}

    try:
        index = get_index()

        results = index.search(
            namespace="siliconmind_docs",
            query={"top_k": n_results, "inputs": {"text": query}}
        )

        hits = results.get("result", {}).get("hits", [])

        if not hits:
            return "", [], {
                "level": "none",
                "label": "No relevant documents found",
                "score": 0.0,
                "from_docs": False
            }

        # Best similarity score
        best_score = round(hits[0].get("_score", 0.0), 3)

        # Confidence thresholds (same as ChromaDB retrieval)
        if best_score >= 0.75:
            level = "high"
            label = f"High confidence — answer grounded in your documents (similarity {best_score})"
        elif best_score >= 0.50:
            level = "medium"
            label = f"Medium confidence — partial match in your documents (similarity {best_score})"
        elif best_score >= 0.30:
            level = "low"
            label = f"Low confidence — weak match; answer may use general knowledge (similarity {best_score})"
        else:
            level = "none"
            label = "Answering from general knowledge — no relevant match in your documents"

        confidence = {
            "level":     level,
            "label":     label,
            "score":     best_score,
            "from_docs": best_score >= 0.50
        }

        # Build context
        sources = list({h.get("source", "Unknown") for h in hits})
        context_parts = []
        for hit in hits:
            src    = hit.get("source", "Unknown")
            topics = hit.get("topics", "")
            text   = hit.get("text", hit.get("_id", ""))
            score  = round(hit.get("_score", 0.0), 3)
            context_parts.append(
                f"[Source: {src} | Topic: {topics} | Similarity: {score}]\n{text}"
            )

        return "\n\n---\n\n".join(context_parts), sources, confidence

    except Exception as e:
        return "", [], {
            "level": "none",
            "label": f"Search error: {e}",
            "score": 0.0,
            "from_docs": False
        }


def get_doc_count() -> int:
    """Return number of vectors stored in Pinecone."""
    try:
        index = get_index()
        stats = index.describe_index_stats()
        ns = stats.get("namespaces", {}).get("siliconmind_docs", {})
        return ns.get("vector_count", 0)
    except Exception:
        return 0