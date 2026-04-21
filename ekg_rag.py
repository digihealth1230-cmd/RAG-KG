"""
ekg_rag.py

EKG-RAG: Evidence and Knowledge-Grounded Reformulation Pipeline.

Implements Section 3.2 of the paper. Given an input consumer health
question q, the pipeline:

  1. Retrieves evidence E via hybrid sparse-dense retrieval over
     PubMed abstracts and MedlinePlus articles (SPLADE-v2 + SapBERT re-rank + MMR)
  2. Constructs structured knowledge K via SapBERT entity linking +
     UMLS ontology-guided subgraph retrieval
  3. Formats a grounded prompt and calls the backbone LLM to produce q_hat
"""

import os
import sys
import re
from typing import List, Dict, Optional, Tuple
import numpy as np

from retrieval.splade_retriever import SPLADERetriever, mmr_select
from retrieval.dense_reranker import rerank, format_evidence_block
from sapbert_extractor import concept_embeddings_for_kg
from kg_builder import load_umls_triples, build_subgraph, format_kg_context

# ---------------------------------------------------------------------------
# Prompt template (Figure 3 in the paper)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = "You are a medical question reformulation specialist."

_USER_TEMPLATE = """\
Objective: Convert the verbose consumer health question below into a \
concise professional medical question.

Clinical Requirements:
- Preserve all symptoms, conditions, diagnoses, medications, procedures, \
negation markers, uncertainty expressions, and follow-up intent.
- Do not introduce claims not present in the original question or evidence.
- Be concise but clinically complete.

Evidence Context:
«BEGIN»
{evidence}
«END»

Knowledge Context:
{knowledge}

Input Question: {question}

Reformulated Question:"""


def _build_prompt(question: str, evidence: str, knowledge: str) -> Tuple[str, str]:
    """Return (system_prompt, user_prompt) for chat-template formatting."""
    user = _USER_TEMPLATE.format(
        evidence=evidence,
        knowledge=knowledge,
        question=question,
    )
    return _SYSTEM_PROMPT, user


# ---------------------------------------------------------------------------
# EKG-RAG pipeline class
# ---------------------------------------------------------------------------

class EKGRag:
    """
    End-to-end EKG-RAG reformulation pipeline.

    Usage:
        pipeline = EKGRag(retriever, umls_index)
        q_hat = pipeline.reformulate(q, model_fn)
    """

    def __init__(
        self,
        retriever: SPLADERetriever,
        umls_index: Optional[Dict] = None,
        token_budget: int = 512,
        mmr_lambda: float = 0.7,
        top_k_sparse: int = 20,
        top_k_dense: int = 10,
        max_kg_triples: int = 15,
        use_evidence: bool = True,
        use_kg: bool = True,
    ):
        self.retriever = retriever
        self.umls_index = umls_index
        self.token_budget = token_budget
        self.mmr_lambda = mmr_lambda
        self.top_k_sparse = top_k_sparse
        self.top_k_dense = top_k_dense
        self.max_kg_triples = max_kg_triples
        self.use_evidence = use_evidence
        self.use_kg = use_kg

    def _get_evidence(self, question: str) -> str:
        """Run sparse-dense retrieval + MMR and format the evidence block."""
        if not self.use_evidence:
            return "(retrieval disabled)"

        # Step 1: SPLADE sparse retrieval with section priors
        candidates = self.retriever.retrieve(question, top_k=self.top_k_sparse)
        if not candidates:
            return "(no evidence retrieved)"

        # Step 2: SapBERT dense re-ranking
        reranked = rerank(question, candidates, top_k=self.top_k_dense)

        # Step 3: Budgeted MMR diversification
        if reranked:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
            texts = [t for t, _ in reranked]
            embs = model.encode(texts, normalize_embeddings=True)
            selected = mmr_select(
                reranked, embs,
                token_budget=self.token_budget,
                lambda_mmr=self.mmr_lambda,
            )
        else:
            selected = [t for t, _ in candidates[:5]]

        return format_evidence_block(selected)

    def _get_knowledge(self, question: str) -> str:
        """Build input-conditioned UMLS subgraph and format as K context."""
        if not self.use_kg:
            return "(knowledge grounding disabled)"

        spans, embs = concept_embeddings_for_kg(question)
        if embs.shape[0] == 0:
            return "(no clinical concepts extracted)"

        triples = build_subgraph(
            self.umls_index,
            embs,
            spans,
            max_triples=self.max_kg_triples,
        )
        return format_kg_context(triples)

    def build_prompt(self, question: str) -> Tuple[str, str]:
        """
        Build (system_prompt, user_prompt) for a given question.
        Does NOT call the model — useful for batched or async inference.
        """
        evidence = self._get_evidence(question)
        knowledge = self._get_knowledge(question)
        return _build_prompt(question, evidence, knowledge)

    def reformulate(
        self,
        question: str,
        model_fn,
    ) -> str:
        """
        Reformulate `question` using the provided `model_fn`.

        `model_fn` signature: (system_prompt: str, user_prompt: str) -> str
        """
        sys_prompt, usr_prompt = self.build_prompt(question)
        return model_fn(sys_prompt, usr_prompt)

    def reformulate_batch(
        self,
        questions: List[str],
        model_fn,
        batch_size: int = 8,
    ) -> List[str]:
        """
        Reformulate a list of questions in mini-batches.
        `model_fn` accepts (list_of_system_prompts, list_of_user_prompts) -> list_of_strings.
        """
        results = []
        for i in range(0, len(questions), batch_size):
            batch = questions[i : i + batch_size]
            prompts = [self.build_prompt(q) for q in batch]
            sys_batch = [p[0] for p in prompts]
            usr_batch = [p[1] for p in prompts]
            batch_out = model_fn(sys_batch, usr_batch)
            results.extend(batch_out)
        return results


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_ekg_rag_pipeline(
    corpus_path: Optional[str] = None,
    corpus_passages: Optional[List[str]] = None,
    umls_csv_path: Optional[str] = None,
    splade_index_path: Optional[str] = None,
    **kwargs,
) -> EKGRag:
    """
    Build a ready-to-use EKGRag instance.

    Either `corpus_path` (a .txt file with one passage per line) or
    `corpus_passages` (a list of strings) must be provided.
    `umls_csv_path` is optional but recommended for full KG grounding.
    """
    # Load passages
    if corpus_passages is None:
        if corpus_path is None:
            raise ValueError("Provide corpus_path or corpus_passages.")
        with open(corpus_path, encoding="utf-8") as f:
            corpus_passages = [line.strip() for line in f if line.strip()]

    # Build / load SPLADE index
    retriever = SPLADERetriever(use_dense_fallback=True)
    if splade_index_path and os.path.exists(splade_index_path):
        retriever.load_index(splade_index_path, corpus_passages)
    else:
        retriever.build_index(corpus_passages)

    # Load UMLS triples if available
    umls_index = None
    if umls_csv_path:
        umls_index = load_umls_triples(umls_csv_path)

    return EKGRag(retriever, umls_index, **kwargs)


# ---------------------------------------------------------------------------
# HuggingFace model wrapper (used in evaluation)
# ---------------------------------------------------------------------------

def make_hf_model_fn(model, tokenizer, max_new_tokens: int = 128, device: str = "cuda"):
    """
    Wrap a HuggingFace causal LM as a `model_fn` accepted by EKGRag.
    Handles both single-prompt and batch calls.
    """
    import torch

    def _single(system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _batch(system_prompts: List[str], user_prompts: List[str]) -> List[str]:
        return [_single(s, u) for s, u in zip(system_prompts, user_prompts)]

    # Return a callable that handles both signatures
    def _model_fn(system_prompt, user_prompt):
        if isinstance(system_prompt, list):
            return _batch(system_prompt, user_prompt)
        return _single(system_prompt, user_prompt)

    return _model_fn


if __name__ == "__main__":
    # Smoke test: build pipeline on a tiny corpus and print the prompt
    tiny_corpus = [
        "Percutaneous nephrolithotomy (PCNL) is a minimally invasive surgical procedure "
        "for removing large kidney stones via a small incision in the back.",
        "Nephrostomy is a procedure in which a tube (nephrostomy tube) is inserted through "
        "the skin into the kidney to drain urine directly from the kidney.",
        "Indications: PCNL is indicated for kidney stones >2cm or stones resistant to ESWL.",
        "Contraindications: uncorrected coagulopathy is a contraindication for PCNL.",
        "Follow-up: post-PCNL patients should attend a urology clinic at 4 weeks.",
    ]

    pipeline = build_ekg_rag_pipeline(corpus_passages=tiny_corpus)

    test_q = (
        "I am trying to get travel insurance and the insurance company don't "
        "recognise the condition by this name or nephrostomy — is there a name "
        "that would be recognised by the insurance company?"
    )

    sys_p, usr_p = pipeline.build_prompt(test_q)
    print("=== SYSTEM PROMPT ===")
    print(sys_p)
    print("\n=== USER PROMPT (first 600 chars) ===")
    print(usr_p[:600])
