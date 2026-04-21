#!/usr/bin/env python3
"""
demo.py

End-to-end demo of EKG-RAG reformulation on a single example.
Shows the full pipeline from question → evidence retrieval → KG grounding → reformulation.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from ekg_rag import build_ekg_rag_pipeline, make_hf_model_fn
from medfaith_f1 import medfaith_f1_single, print_medfaith_report


def main():
    # -------------------------------------------------------------------------
    # 1. Setup: tiny demo corpus (in production, load from file)
    # -------------------------------------------------------------------------
    demo_corpus = [
        "Percutaneous nephrolithotomy (PCNL) is a minimally invasive surgical procedure "
        "for removing large kidney stones via a small incision in the back.",
        "Nephrostomy is a procedure in which a tube (nephrostomy tube) is inserted through "
        "the skin into the kidney to drain urine directly from the kidney.",
        "Indications: PCNL is indicated for kidney stones >2cm or stones resistant to ESWL.",
        "Contraindications: uncorrected coagulopathy is a contraindication for PCNL.",
        "Follow-up: post-PCNL patients should attend a urology clinic at 4 weeks for imaging.",
        "Alternative names: percutaneous nephrolithotomy is also called percutaneous kidney stone removal.",
    ]

    # -------------------------------------------------------------------------
    # 2. Build EKG-RAG pipeline (without KG for this demo — set umls_csv_path if available)
    # -------------------------------------------------------------------------
    print("[1/4] Building EKG-RAG pipeline...")
    pipeline = build_ekg_rag_pipeline(
        corpus_passages=demo_corpus,
        umls_csv_path=None,  # No KG in this demo; pass path to enable full EKG-RAG
        token_budget=300,
        mmr_lambda=0.7,
    )
    print("      Pipeline ready.\n")

    # -------------------------------------------------------------------------
    # 3. Example consumer health question (from MeQSum)
    # -------------------------------------------------------------------------
    question = (
        "Subject: percutaneous nephrolithotomy. Message: I am trying to get travel "
        "insurance and the insurance company don't recognise the condition by this name "
        "or nephrostomy is there a name that would be recognised by the insurance company?"
    )

    expert_reference = (
        "What are other names for percutaneous nephrolithotomy or nephrostomy?"
    )

    print("[2/4] Input question:")
    print(f"      {question}\n")

    # -------------------------------------------------------------------------
    # 4. Build grounded prompt (no model call yet)
    # -------------------------------------------------------------------------
    print("[3/4] Building grounded prompt with evidence and knowledge...")
    sys_prompt, usr_prompt = pipeline.build_prompt(question)

    print("\n--- SYSTEM PROMPT ---")
    print(sys_prompt)
    print("\n--- USER PROMPT (truncated) ---")
    print(usr_prompt[:600], "...\n")

    # -------------------------------------------------------------------------
    # 5. Mock reformulation (replace with actual model call in production)
    # -------------------------------------------------------------------------
    # In real use:
    #   from transformers import AutoTokenizer, AutoModelForCausalLM
    #   tok, model = ...
    #   model_fn = make_hf_model_fn(model, tok)
    #   reformulation = pipeline.reformulate(question, model_fn)

    # Mock output for demo
    reformulation_mock = "What are the alternative names for percutaneous nephrolithotomy?"

    print("[4/4] Reformulation (mocked):")
    print(f"      {reformulation_mock}\n")

    # -------------------------------------------------------------------------
    # 6. Evaluate with MedFaith-F1
    # -------------------------------------------------------------------------
    print("--- MedFaith-F1 Evaluation ---")
    scores = medfaith_f1_single(question, reformulation_mock)
    for k, v in scores.items():
        print(f"  {k:15s}: {v:.4f}")

    print("\n--- Notes ---")
    print("This demo uses a mock reformulation. To run with a real model:")
    print("  1. Load a HuggingFace model: tok, model = load_model_and_tokenizer(...)")
    print("  2. Wrap it: model_fn = make_hf_model_fn(model, tok)")
    print("  3. Call: reformulation = pipeline.reformulate(question, model_fn)")
    print("\nFor full evaluation across datasets, use evaluation/run_eval.py")


if __name__ == "__main__":
    main()
