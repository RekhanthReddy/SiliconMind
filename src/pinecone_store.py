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

    batch_size = 50
    for i in range(0, len(records), batch_size):
        index.upsert_records(
            namespace="siliconmind_docs",
            records=records[i:i + batch_size]
        )
    return len(records)


def search_chunks(query: str, n_results: int = 5) -> tuple[str, list[str], dict]:
    if not query:
        return "", [], {"level": "none", "label": "No query provided", "score": 0.0, "from_docs": False}

    try:
        index = get_index()

        results = index.search(
            namespace="siliconmind_docs",
            query={"top_k": n_results, "inputs": {"text": query}}
        )

        # Handle both dict and object response formats
        if hasattr(results, 'result'):
            hits = results.result.hits
        else:
            hits = results.get("result", {}).get("hits", [])

        if not hits:
            return "", [], {
                "level": "none",
                "label": "No relevant documents found",
                "score": 0.0,
                "from_docs": False
            }

        # Best similarity score
        first = hits[0]
        best_score = round(first._score if hasattr(first, '_score') else first.get("_score", 0.0), 3)

        # Confidence thresholds
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
        context_parts = []
        sources_set = set()

        for hit in hits:
            fields = hit.fields if hasattr(hit, 'fields') else hit
            if isinstance(fields, dict):
                src    = fields.get("source", "Unknown")
                topics = fields.get("topics", "")
                text   = fields.get("text", "")
            else:
                src    = getattr(fields, "source", "Unknown")
                topics = getattr(fields, "topics", "")
                text   = getattr(fields, "text", "")

            score = round(hit._score if hasattr(hit, '_score') else hit.get("_score", 0.0), 3)
            sources_set.add(src)
            context_parts.append(
                f"[Source: {src} | Topic: {topics} | Similarity: {score}]\n{text}"
            )

        return "\n\n---\n\n".join(context_parts), list(sources_set), confidence

    except Exception as e:
        return "", [], {
            "level": "none",
            "label": f"Search error: {e}",
            "score": 0.0,
            "from_docs": False
        }


def get_doc_count() -> int:
    try:
        index = get_index()
        stats = index.describe_index_stats()
        ns = stats.get("namespaces", {}).get("siliconmind_docs", {})
        return ns.get("vector_count", 0)
    except Exception:
        return 0