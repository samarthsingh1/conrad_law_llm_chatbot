# src/evaluation/retrieval_metrics.py

def _extract_chunk_id(doc):
    """Safely extract chunk_id from a LangChain Document.metadata"""
    meta = getattr(doc, "metadata", {})
    return meta.get("chunk_id")


# ---------------------------------------------------------
# RETRIEVAL METRICS
# ---------------------------------------------------------

def recall_at_k(retrieved_docs, relevant_ids, k):
    """Recall@K = (# relevant retrieved) / (# total relevant)."""
    retrieved_ids = [_extract_chunk_id(d) for d in retrieved_docs[:k]]
    retrieved_ids = [x for x in retrieved_ids if x is not None]

    hits = sum(1 for rid in retrieved_ids if rid in relevant_ids)
    total_relevant = len(relevant_ids)

    if total_relevant == 0:
        return 0.0
    return hits / total_relevant


def precision_at_k(retrieved_docs, relevant_ids, k):
    """Precision@K = (# relevant retrieved) / K."""
    retrieved_ids = [_extract_chunk_id(d) for d in retrieved_docs[:k]]
    retrieved_ids = [x for x in retrieved_ids if x is not None]

    hits = sum(1 for rid in retrieved_ids if rid in relevant_ids)
    return hits / k


def hit_rate_at_k(retrieved_docs, relevant_ids, k):
    """Hit@K = 1 if any relevant doc appears in top K, else 0."""
    retrieved_ids = [_extract_chunk_id(d) for d in retrieved_docs[:k]]
    retrieved_ids = [x for x in retrieved_ids if x is not None]

    return 1.0 if any(rid in relevant_ids for rid in retrieved_ids) else 0.0


def reciprocal_rank(retrieved_docs, relevant_ids):
    """
    MRR = 1 / rank_of_first_relevant
    rank is 1-indexed
    """
    for idx, doc in enumerate(retrieved_docs):
        cid = _extract_chunk_id(doc)
        if cid in relevant_ids:
            return 1.0 / (idx + 1)
    return 0.0
