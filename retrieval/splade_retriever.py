"""
retrieval/splade_retriever.py

Sparse retrieval using SPLADE-v2 with section-based scoring and
token-budgeted Maximal Marginal Relevance (MMR) diversification.

Used as the first leg of EKG-RAG evidence retrieval (E channel).
"""

import re
from typing import List, Tuple, Dict, Optional
import numpy as np

# Section priors — up-weight clinically informative regions
_SECTION_PRIORS: Dict[str, float] = {
    "indication": 1.5,
    "indications": 1.5,
    "contraindication": 1.4,
    "contraindications": 1.4,
    "side effect": 1.3,
    "adverse": 1.3,
    "treatment": 1.3,
    "diagnosis": 1.4,
    "procedure": 1.3,
    "follow-up": 1.2,
    "dosage": 1.2,
    "mechanism": 1.1,
    "overview": 1.0,
    "description": 1.0,
}

_SECTION_RE = re.compile(
    r"^\s*(" + "|".join(re.escape(k) for k in _SECTION_PRIORS) + r")\s*[:\-]",
    re.IGNORECASE | re.MULTILINE,
)


def _section_weight(passage: str) -> float:
    """Return the highest section prior found in a passage (default 1.0)."""
    best = 1.0
    for m in _SECTION_RE.finditer(passage):
        label = m.group(1).lower()
        for key, w in _SECTION_PRIORS.items():
            if key in label:
                best = max(best, w)
    return best


def _approx_token_count(text: str) -> int:
    """Rough whitespace-based token count. Not BPE but sufficient for budgeting."""
    return len(text.split())


class SPLADERetriever:
    """
    Wraps a SPLADE-v2 model for sparse retrieval over a pre-built passage index.

    In a live deployment the index is a FAISS-backed inverted index of SPLADE
    sparse vectors. For reproducibility the class also supports a simple
    cosine-over-dense fallback when the full SPLADE setup is unavailable.
    """

    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        index_path: Optional[str] = None,
        use_dense_fallback: bool = True,
    ):
        self.model_name = model_name
        self.index_path = index_path
        self.use_dense_fallback = use_dense_fallback
        self._model = None
        self._index = None  # populated by build_index() or load_index()

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            import torch

            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self._model.eval()
            self._device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
            self._model.to(self._device)
        except Exception as e:
            if self.use_dense_fallback:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
                )
                self._device = "cpu"
                self._is_fallback = True
            else:
                raise RuntimeError(f"Could not load SPLADE model: {e}")

    def _encode_splade(self, text: str) -> np.ndarray:
        """Encode text → sparse SPLADE vector (dense array, mostly zeros)."""
        import torch

        inputs = self._tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        ).to(self._device)
        with torch.no_grad():
            out = self._model(**inputs)
        # SPLADE aggregation: relu(log(1 + relu(logits))) max-pooled over tokens
        logits = out.logits  # (1, seq_len, vocab_size)
        vec = torch.log1p(torch.relu(logits)).max(dim=1).values.squeeze(0)
        return vec.cpu().numpy()

    def build_index(self, passages: List[str]) -> None:
        """
        Build a sparse (or dense fallback) index over `passages`.
        Stores (passage, sparse_vec, section_weight) per entry.
        """
        self._load_model()
        self._passages = passages
        self._section_weights = np.array([_section_weight(p) for p in passages])

        if getattr(self, "_is_fallback", False):
            self._index = self._model.encode(passages, normalize_embeddings=True)
        else:
            self._index = np.vstack([self._encode_splade(p) for p in passages])

    def load_index(self, index_path: str, passages: List[str]) -> None:
        """Load a pre-built numpy index from disk."""
        self._passages = passages
        self._index = np.load(index_path)
        self._section_weights = np.array([_section_weight(p) for p in passages])

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-K passages for `query`.
        Returns [(passage_text, score), ...] sorted descending.
        """
        self._load_model()
        if self._index is None:
            raise RuntimeError("Index not built. Call build_index() or load_index() first.")

        if getattr(self, "_is_fallback", False):
            q_emb = self._model.encode([query], normalize_embeddings=True)
            raw_scores = (self._index @ q_emb.T).squeeze()
        else:
            q_vec = self._encode_splade(query)
            # Dot product for sparse SPLADE
            raw_scores = self._index @ q_vec

        # Apply section priors
        scores = raw_scores * self._section_weights
        top_idx = np.argsort(-scores)[:top_k]
        return [(self._passages[i], float(scores[i])) for i in top_idx]


def mmr_select(
    candidates: List[Tuple[str, float]],
    candidate_embeddings: np.ndarray,
    token_budget: int = 512,
    lambda_mmr: float = 0.7,
) -> List[str]:
    """
    Budgeted Maximal Marginal Relevance selection from a ranked candidate list.

    MMR(p_i, q) = λ·sim(p_i, q) - (1-λ)·max_{p_j in S} sim(p_i, p_j)

    Selects passages greedily until the token budget is exhausted.

    Args:
        candidates:           [(passage_text, relevance_score), ...]
        candidate_embeddings: (n_cands, dim) dense embeddings for all candidates
        token_budget:         max total tokens in selected set
        lambda_mmr:           relevance-diversity tradeoff (λ=1 → pure relevance)

    Returns:
        selected passage texts
    """
    n = len(candidates)
    if n == 0:
        return []

    relevance = np.array([s for _, s in candidates])
    # Normalise to [0,1] for comparability
    rel_range = relevance.max() - relevance.min()
    if rel_range > 0:
        relevance = (relevance - relevance.min()) / rel_range

    # Cosine similarity matrix
    norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-9
    normed = candidate_embeddings / norms
    sim_matrix = normed @ normed.T  # (n, n)

    selected_idx: List[int] = []
    remaining_idx = list(range(n))
    budget_used = 0

    while remaining_idx:
        if not selected_idx:
            # First selection: pure relevance
            best_i = max(remaining_idx, key=lambda i: relevance[i])
        else:
            # MMR score
            sel_embs = normed[selected_idx]  # (|S|, dim)

            def _mmr(i):
                rel_score = lambda_mmr * relevance[i]
                max_sim_to_sel = float(sim_matrix[i][selected_idx].max())
                div_penalty = (1.0 - lambda_mmr) * max_sim_to_sel
                return rel_score - div_penalty

            best_i = max(remaining_idx, key=_mmr)

        passage_text = candidates[best_i][0]
        tok_count = _approx_token_count(passage_text)

        if budget_used + tok_count > token_budget and selected_idx:
            # Budget would be exceeded — still try smaller remaining candidates
            remaining_idx.remove(best_i)
            # Try a shorter one if available
            fallback = [
                i for i in remaining_idx
                if budget_used + _approx_token_count(candidates[i][0]) <= token_budget
            ]
            if not fallback:
                break
            best_i = max(fallback, key=lambda i: relevance[i])
            passage_text = candidates[best_i][0]
            tok_count = _approx_token_count(passage_text)

        selected_idx.append(best_i)
        remaining_idx = [i for i in remaining_idx if i != best_i]
        budget_used += tok_count

        if budget_used >= token_budget:
            break

    return [candidates[i][0] for i in selected_idx]


if __name__ == "__main__":
    # Smoke test with tiny synthetic corpus
    corpus = [
        "Indications: Percutaneous nephrolithotomy is used to remove kidney stones larger than 2cm.",
        "Contraindications: Patients on anticoagulants may need dose adjustment before the procedure.",
        "Nephrostomy tube placement provides temporary urinary drainage after pyeloplasty.",
        "Follow-up: A urine culture should be obtained 4-6 weeks post-procedure.",
        "Overview of minimally invasive urological procedures in adults.",
    ]

    retriever = SPLADERetriever(use_dense_fallback=True)
    retriever.build_index(corpus)
    results = retriever.retrieve("percutaneous nephrolithotomy alternative names", top_k=5)
    print("Top-5 retrieved:")
    for text, score in results:
        print(f"  [{score:.3f}] {text[:80]}")
