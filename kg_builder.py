"""
kg_builder.py

Ontology-guided knowledge graph construction for EKG-RAG.
Builds an input-conditioned subgraph from UMLS concept relations
normalized via SNOMED (Dx, Proc) and RxNorm (Rx).

Mirrors the K channel described in Section 3.2 of the paper.
"""

import os
import csv
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
from sentence_transformers import SentenceTransformer

_SAPBERT_MODEL: Optional[SentenceTransformer] = None

# Relations retained in the subgraph — clinically meaningful only
_SALIENT_RELATIONS: Set[str] = {
    "finding_of",
    "treats",
    "contraindicated_with",
    "associated_with",
    "is_a",
    "part_of",
    "causative_agent_of",
    "has_active_ingredient",
    "indicated_for",
    "has_finding_site",
}

# SNOMED top-level semantic types we care about
_SNOMED_DX_PROC_TYPES = {"disorder", "procedure", "finding", "body structure"}
_RXNORM_TYPES = {"clinical drug", "ingredient", "drug"}


def _get_sapbert() -> SentenceTransformer:
    global _SAPBERT_MODEL
    if _SAPBERT_MODEL is None:
        _SAPBERT_MODEL = SentenceTransformer(
            "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        )
    return _SAPBERT_MODEL


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------

def load_umls_triples(csv_path: str) -> Optional[Dict]:
    """
    Load UMLS triples from a CSV file with columns: head, relation, tail.

    Builds two indices:
      - triples:  list of (head, relation, tail)
      - by_node:  dict[node_str] -> list of triples involving that node

    CSV may use 'subject/predicate/object' column names as fallback.
    """
    if not os.path.exists(csv_path):
        return None

    triples: List[Tuple[str, str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        raw_header = next(reader)
        cols = [h.strip().lower() for h in raw_header]

        def _col(row, primary, fallback):
            if primary in cols:
                return row[cols.index(primary)].strip()
            if fallback in cols:
                return row[cols.index(fallback)].strip()
            return ""

        for row in reader:
            if len(row) < 3:
                continue
            h = _col(row, "head", "subject")
            r = _col(row, "relation", "predicate") or "associated_with"
            t = _col(row, "tail", "object")
            if h and t:
                triples.append((h, r, t))

    by_node: Dict[str, List[Tuple[str, str, str]]] = {}
    for h, r, t in triples:
        for node in (h.lower(), t.lower()):
            by_node.setdefault(node, []).append((h, r, t))

    return {"triples": triples, "by_node": by_node}


# ---------------------------------------------------------------------------
# Input-conditioned subgraph retrieval
# ---------------------------------------------------------------------------

def build_subgraph(
    umls_index: Optional[Dict],
    entity_embeddings: np.ndarray,
    entity_spans: List[str],
    max_triples: int = 15,
    sim_threshold: float = 0.70,
) -> List[Tuple[str, str, str]]:
    """
    Given SapBERT embeddings of input entities, retrieve the most relevant
    UMLS triples by cosine similarity to indexed node embeddings.

    Args:
        umls_index:         output of load_umls_triples()
        entity_embeddings:  (n_entities, dim) SapBERT embeddings of spans
        entity_spans:       corresponding span strings
        max_triples:        hard cap on subgraph size
        sim_threshold:      minimum cosine similarity to include a node

    Returns:
        list of (head, relation, tail) triples
    """
    if umls_index is None or entity_embeddings.shape[0] == 0:
        return []

    sapbert = _get_sapbert()
    node_keys = list(umls_index["by_node"].keys())
    if not node_keys:
        return []

    node_embs = sapbert.encode(node_keys, normalize_embeddings=True)
    ent_embs_norm = entity_embeddings / (
        np.linalg.norm(entity_embeddings, axis=1, keepdims=True) + 1e-9
    )
    sim = ent_embs_norm @ node_embs.T  # (n_ents, n_nodes)

    top_node_idx = np.unique(np.where(sim >= sim_threshold)[1])
    if len(top_node_idx) == 0:
        # Fallback: take top-3 per entity regardless of threshold
        top_node_idx = np.unique(np.argsort(-sim, axis=1)[:, :3].ravel())

    subgraph: List[Tuple[str, str, str]] = []
    seen: Set[Tuple[str, str, str]] = set()

    # Sort nodes by max similarity score (most relevant first)
    node_max_sim = sim[:, top_node_idx].max(axis=0)
    sorted_idx = top_node_idx[np.argsort(-node_max_sim)]

    for idx in sorted_idx:
        node_key = node_keys[idx]
        for triple in umls_index["by_node"].get(node_key, []):
            if triple not in seen and triple[1] in _SALIENT_RELATIONS:
                subgraph.append(triple)
                seen.add(triple)
                if len(subgraph) >= max_triples:
                    return subgraph

    return subgraph


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_kg_context(triples: List[Tuple[str, str, str]]) -> str:
    """
    Format knowledge graph triples into the K context block for the prompt.
    """
    if not triples:
        return "(no knowledge graph context)"
    lines = [f"• ({h}) --[{r}]--> ({t})" for h, r, t in triples]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Normalisation helpers (SNOMED / RxNorm stubs)
# ---------------------------------------------------------------------------

def snomed_normalise(span: str) -> str:
    """
    Map a clinical span to its preferred SNOMED-CT term.
    In production: query the SNOMED REST API or a pre-loaded concept table.
    Returns the input span unchanged if no mapping found.
    """
    # Stub — extend with actual SNOMED REST call or local table
    _snomed_aliases = {
        "nephrostomy": "nephrostomy (procedure)",
        "nephrolithotomy": "percutaneous nephrolithotomy (procedure)",
        "diabetes": "diabetes mellitus (disorder)",
        "hypertension": "essential hypertension (disorder)",
        "pneumonia": "pneumonia (disorder)",
    }
    return _snomed_aliases.get(span.lower(), span)


def rxnorm_normalise(span: str) -> str:
    """
    Map a medication span to its preferred RxNorm term.
    In production: query the RxNorm REST API.
    Returns the input span unchanged if no mapping found.
    """
    _rxnorm_aliases = {
        "aspirin": "aspirin (rxnorm:1191)",
        "metformin": "metformin (rxnorm:6809)",
        "furosemide": "furosemide (rxnorm:4603)",
        "lasix": "furosemide (rxnorm:4603)",
        "warfarin": "warfarin (rxnorm:11289)",
    }
    return _rxnorm_aliases.get(span.lower(), span)


def normalise_span(span: str, category: str) -> str:
    """Route normalisation by category."""
    if category in ("Dx", "Proc"):
        return snomed_normalise(span)
    elif category == "Rx":
        return rxnorm_normalise(span)
    return span


if __name__ == "__main__":
    from sapbert_extractor import concept_embeddings_for_kg

    query = (
        "I am trying to get travel insurance and the insurance company "
        "don't recognise percutaneous nephrolithotomy or nephrostomy."
    )

    spans, embs = concept_embeddings_for_kg(query)
    print(f"Extracted {len(spans)} entity spans: {spans[:8]}")

    # Without a real UMLS CSV, demonstrate fallback
    subgraph = build_subgraph(
        umls_index=None,
        entity_embeddings=embs,
        entity_spans=spans,
    )
    print(f"Subgraph triples (no index): {subgraph}")
    print("KG context:", format_kg_context(subgraph))
