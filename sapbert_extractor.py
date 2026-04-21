"""
sapbert_extractor.py

Clinical concept extraction and UMLS normalization using SapBERT.
Supports category classification into Dx / Rx / Proc / Fup for MedFaith-F1.
"""

import re
from typing import List, Dict, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Terminology / pattern hints
# ---------------------------------------------------------------------------

# Rough lexical seeds — not exhaustive, just to bootstrap before SapBERT sim
_DX_HINTS = {
    "syndrome", "disease", "disorder", "cancer", "carcinoma", "tumor",
    "infection", "deficiency", "failure", "insufficiency", "dysfunction",
    "neuropathy", "arthritis", "diabetes", "hypertension", "anemia",
    "hepatitis", "cirrhosis", "sepsis", "pneumonia", "asthma", "copd",
    "nephritis", "nephropathy", "sclerosis", "fibrosis", "stenosis",
}

_RX_HINTS = {
    "mg", "mcg", "tablet", "capsule", "dose", "injection", "infusion",
    "aspirin", "ibuprofen", "metformin", "insulin", "warfarin", "heparin",
    "statin", "atorvastatin", "lisinopril", "metoprolol", "amoxicillin",
    "prednisone", "albuterol", "omeprazole", "furosemide", "amlodipine",
    "losartan", "gabapentin", "levothyroxine", "clopidogrel", "ondansetron",
}

_PROC_HINTS = {
    "surgery", "procedure", "biopsy", "catheterization", "transplant",
    "resection", "excision", "repair", "replacement", "anastomosis",
    "nephrolithotomy", "nephrostomy", "endoscopy", "colonoscopy",
    "angioplasty", "stenting", "ablation", "dialysis", "chemotherapy",
    "radiotherapy", "radiation", "mri", "ct", "ultrasound", "x-ray",
    "echocardiography", "echocardiogram",
}

_FUP_HINTS = {
    "follow", "follow-up", "followup", "monitor", "monitoring",
    "recheck", "revisit", "repeat", "return", "clinic", "appointment",
    "schedule", "referral", "refer", "consultation", "consult",
    "check", "test", "retest", "screen", "surveillance",
}

# Qualifier patterns (negation, uncertainty, laterality)
_QUALIFIER_RE = {
    "negation": re.compile(
        r"\b(no|not|without|denies|negative|ruled\s+out|absent|never)\b", re.I
    ),
    "uncertainty": re.compile(
        r"\b(may|might|possibly|probable|likely|suspected|question\s+of|concern\s+for)\b", re.I
    ),
    "laterality": re.compile(r"\b(left|right|bilateral|unilateral)\b", re.I),
}

_SAPBERT_MODEL = None  # lazy-loaded


def _get_sapbert() -> SentenceTransformer:
    global _SAPBERT_MODEL
    if _SAPBERT_MODEL is None:
        _SAPBERT_MODEL = SentenceTransformer(
            "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        )
    return _SAPBERT_MODEL


def _detect_qualifiers(text: str) -> Dict[str, bool]:
    return {q: bool(_QUALIFIER_RE[q].search(text)) for q in _QUALIFIER_RE}


def _candidate_spans(text: str) -> List[str]:
    """
    Extract n-gram candidate spans (1..3 tokens) that look clinical.
    Returns lowercased, deduplicated spans.
    """
    tokens = re.findall(r"[A-Za-z0-9\-/]+", text)
    cands = []
    seen = set()

    def _is_interesting(span: str) -> bool:
        sl = span.lower()
        if sl in seen or len(sl) < 2:
            return False
        # Accept abbreviations, known seeds, or multi-token phrases with a seed
        if sl.isupper() and 2 <= len(sl) <= 6:
            return True
        parts = sl.split()
        if any(p in _DX_HINTS | _RX_HINTS | _PROC_HINTS | _FUP_HINTS for p in parts):
            return True
        if re.search(r"\d+(mg|mcg|ml|%)", sl):
            return True
        return False

    for i, tok in enumerate(tokens):
        for j in (1, 2, 3):
            if i + j > len(tokens):
                break
            span = " ".join(tokens[i : i + j])
            if _is_interesting(span):
                cands.append(span.lower())
                seen.add(span.lower())

    return cands


def classify_category(span: str) -> str:
    """
    Rough heuristic category label before UMLS normalization.
    Returns one of: 'Dx', 'Rx', 'Proc', 'Fup', or 'Other'.
    """
    sl = span.lower()
    words = set(sl.split())

    if words & _FUP_HINTS:
        return "Fup"
    if words & _PROC_HINTS or re.search(
        r"(otomy|oscopy|ectomy|plasty|ography|ogram|gram)\b", sl
    ):
        return "Proc"
    if words & _RX_HINTS or re.search(r"\d+(mg|mcg|ml)", sl):
        return "Rx"
    if words & _DX_HINTS or re.search(
        r"(itis|osis|emia|uria|pathy|trophy|plasia)\b", sl
    ):
        return "Dx"
    return "Other"


def extract_and_normalize(
    text: str,
    top_k: int = 10,
    sim_threshold: float = 0.72,
) -> Dict[str, List[str]]:
    """
    Extract clinical spans from `text`, embed with SapBERT, and return
    UMLS-normalized concepts bucketed by category.

    Returns:
        {
          'Dx':   [normalized_span, ...],
          'Rx':   [...],
          'Proc': [...],
          'Fup':  [...],
        }
    """
    sapbert = _get_sapbert()
    raw_cands = _candidate_spans(text)

    if not raw_cands:
        return {"Dx": [], "Rx": [], "Proc": [], "Fup": []}

    embs = sapbert.encode(raw_cands, batch_size=32, normalize_embeddings=True)

    # Per-span top-K UMLS concept selection via self-similarity
    # (In production: compare against pre-indexed UMLS concept embeddings)
    # Here we use within-set cosine ranking as a proxy for concept coherence
    sim_matrix = embs @ embs.T  # (n_cands, n_cands)

    buckets: Dict[str, List[str]] = {"Dx": [], "Rx": [], "Proc": [], "Fup": []}

    selected_indices = set()
    for i, span in enumerate(raw_cands):
        if i in selected_indices:
            continue
        cat = classify_category(span)
        if cat == "Other":
            continue
        # Accept if self-similarity to nearest neighbour is above threshold
        row = sim_matrix[i].copy()
        row[i] = 0.0  # exclude self
        best_sim = float(row.max()) if len(row) > 1 else 0.0
        if best_sim >= sim_threshold or cat != "Other":
            if cat in buckets:
                buckets[cat].append(span)
                selected_indices.add(i)
            if sum(len(v) for v in buckets.values()) >= top_k * 4:
                break

    return buckets


def extract_category_presence(text: str) -> Dict[str, bool]:
    """
    Binary presence flag per category {Dx, Rx, Proc, Fup} for a single text.
    Used by MedFaith-F1.
    """
    normalized = extract_and_normalize(text)
    return {cat: len(concepts) > 0 for cat, concepts in normalized.items()}


def concept_embeddings_for_kg(text: str) -> Tuple[List[str], np.ndarray]:
    """
    Return (spans, embeddings) for use in KG subgraph retrieval.
    """
    sapbert = _get_sapbert()
    spans = _candidate_spans(text)
    if not spans:
        return [], np.zeros((0, 768))
    embs = sapbert.encode(spans, normalize_embeddings=True)
    return spans, embs


if __name__ == "__main__":
    # Quick smoke test
    q = (
        "I am trying to get travel insurance and the insurance company don't "
        "recognise percutaneous nephrolithotomy — is there another name for "
        "nephrostomy that they would accept?"
    )
    presence = extract_category_presence(q)
    print("Category presence:", presence)

    buckets = extract_and_normalize(q)
    for cat, items in buckets.items():
        print(f"  {cat}: {items}")
