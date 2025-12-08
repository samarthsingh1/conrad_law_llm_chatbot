# src/evaluation/retrieval_eval.py

import os
import sys
from pathlib import Path

# Ensure project root is discoverable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from rag_backend import retrieve_docs, load_user_db, load_cuad_db
from retrieval_metrics import (
    recall_at_k,
    precision_at_k,
    hit_rate_at_k,
    reciprocal_rank
)

# ---------------------------------------------------------
# TEST DATA (or load from JSON file)
# ---------------------------------------------------------

TEST_QUERIES = [
    {
        "query": "What is the governing law clause?",
        "relevant_chunk_ids": [12, 13]
    },
    {
        "query": "When does the term renew automatically?",
        "relevant_chunk_ids": [45]
    }
]

K = 5  # evaluate top-5 metrics


# ---------------------------------------------------------
# MAIN EVALUATION ROUTINE
# ---------------------------------------------------------

def run_eval():
    print("\n==============================")
    print("🔍 Running Retrieval Evaluation")
    print("==============================")

    user_db = load_user_db()
    kb_db = load_cuad_db()

    for item in TEST_QUERIES:
        query = item["query"]
        relevance = item["relevant_chunk_ids"]

        print(f"\n------------------------------------")
        print(f"Query: {query}")
        print(f"Relevant Chunk IDs: {relevance}")

        retrieved_user, retrieved_kb = retrieve_docs(query, k=10)

        # Use correct DB based on routing
        retrieved = retrieved_user if retrieved_user else retrieved_kb

        print("\nTop Retrieved Metadata (first 3):")
        for d in retrieved[:3]:
            print(" -", d.metadata)

        r = recall_at_k(retrieved, relevance, k=K)
        p = precision_at_k(retrieved, relevance, k=K)
        h = hit_rate_at_k(retrieved, relevance, k=K)
        mrr = reciprocal_rank(retrieved, relevance)

        print(f"\nRecall@{K}:     {r:.3f}")
        print(f"Precision@{K}:  {p:.3f}")
        print(f"Hit@{K}:        {h:.3f}")
        print(f"MRR:            {mrr:.3f}")


if __name__ == "__main__":
    run_eval()
