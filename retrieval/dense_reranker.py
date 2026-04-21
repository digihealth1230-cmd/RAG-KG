"""
retrieval/dense_reranker.py

SapBERT-based dense re-ranking of candidate passages retrieved by SPLADE.
Produces the final evidence block E for EKG-RAG prompting.
"""

from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

_DENSE_MODEL = None


def _get_dense_model() -> SentenceTransformer:
    global _DENSE_MODEL
    if _DENSE_MODEL is None:
        _DENSE_MODEL = SentenceTransformer(
            "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        )
    return _DENSE_MODEL


def rerank(
    query: str,
    candidates: List[Tuple[str, float]],
    top_k: int = 10,
    alpha: float = 0.5,
) -> List[Tuple[str, float]]:
    """
    Re-rank `candidates` by combining sparse scores with SapBERT cosine scores.

    Combined score = alpha * dense_score + (1 - alpha) * sparse_score_normalised

    Args:
        query:      the input question
        candidates: [(passage_text, sparse_score), ...] from SPLADE
        top_k:      how many passages to keep
        alpha:      weight for dense score (0 = pure sparse, 1 = pure dense)

    Returns:
        [(passage_text, combined_score), ...] sorted descending, length <= top_k
    """
    if not candidates:
        return []

    model = _get_dense_model()
    texts = [c[0] for c in candidates]
    sparse_scores = np.array([c[1] for c in candidates], dtype=float)

    # Normalise sparse scores to [0, 1]
    sp_range = sparse_scores.max() - sparse_scores.min()
    if sp_range > 0:
        sparse_norm = (sparse_scores - sparse_scores.min()) / sp_range
    else:
        sparse_norm = np.ones(len(candidates))

    q_emb = model.encode([query], normalize_embeddings=True)       # (1, d)
    c_embs = model.encode(texts, normalize_embeddings=True)        # (n, d)
    dense_scores = (c_embs @ q_emb.T).squeeze()                    # (n,)

    combined = alpha * dense_scores + (1.0 - alpha) * sparse_norm
    order = np.argsort(-combined)[:top_k]

    return [(texts[i], float(combined[i])) for i in order]


def format_evidence_block(passages: List[str], max_passages: int = 5) -> str:
    """
    Format a list of retrieved passages into the evidence block string
    used in the EKG-RAG prompt template.
    """
    block_parts = []
    for i, p in enumerate(passages[:max_passages], start=1):
        p_clean = p.strip().replace("\n", " ")
        block_parts.append(f"[E{i}] {p_clean}")
    return "\n".join(block_parts) if block_parts else "(no evidence retrieved)"


if __name__ == "__main__":
    from splade_retriever import SPLADERetriever

    corpus = [
        "Indications: Percutaneous nephrolithotomy is used to remove kidney stones larger than 2cm.",
        "Nephrostomy tube placement provides temporary urinary drainage after pyeloplasty.",
        "Follow-up: A urine culture should be obtained 4-6 weeks post-procedure.",
        "Contraindications: Patients on anticoagulants may need dose adjustment before the procedure.",
        "Overview of minimally invasive urological procedures in adults.",
        "Percutaneous nephrolithotomy (PCNL) is also known as percutaneous kidney stone surgery.",
    ]

    query = "percutaneous nephrolithotomy alternative names nephrostomy"

    retriever = SPLADERetriever(use_dense_fallback=True)
    retriever.build_index(corpus)
    candidates = retriever.retrieve(query, top_k=6)

    reranked = rerank(query, candidates, top_k=4)
    print("Re-ranked evidence:")
    for text, score in reranked:
        print(f"  [{score:.3f}] {text[:90]}")

    print("\nFormatted evidence block:")
    print(format_evidence_block([t for t, _ in reranked]))
