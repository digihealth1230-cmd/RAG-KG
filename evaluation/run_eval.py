"""
evaluation/run_eval.py

Full evaluation loop for the EKG-RAG paper.
Covers zero-shot, few-shot, and PEFT settings across:
  - retrieval configs: none (Base), RAG-only, EKG-RAG
  - datasets: MeQSum, MQP
  - backbone models: LLaMA-3-8B-Instruct, Qwen2.5-7B-Instruct

Computes ROUGE-L, BLEU, BERTScore, MedFaith-F1, CHR, and FKGL
and writes results to a JSON file matching Table 2 format.
"""

import os
import re
import sys
import json
import argparse
import pathlib
import warnings
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from medfaith_f1 import medfaith_f1, print_medfaith_report
from ekg_rag import EKGRag, build_ekg_rag_pipeline, make_hf_model_fn

# ---------------------------------------------------------------------------
# Readability: Flesch-Kincaid Grade Level
# ---------------------------------------------------------------------------

def _syllables(word: str) -> int:
    vowel_runs = re.findall(r"[aeiouy]+", word.lower())
    return max(1, len(vowel_runs))


def fkgl(text: str) -> float:
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    words = re.findall(r"\w+", text)
    if not sentences or not words:
        return 0.0
    n_syl = sum(_syllables(w) for w in words)
    asl = len(words) / len(sentences)   # avg sentence length
    asw = n_syl / len(words)            # avg syllables per word
    return 0.39 * asl + 11.8 * asw - 15.59


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_csv_pairs(path: str, src_col: str, tgt_col: str) -> Tuple[List[str], List[str]]:
    import pandas as pd
    df = pd.read_csv(path)[[src_col, tgt_col]].dropna()
    return df[src_col].tolist(), df[tgt_col].tolist()


def load_dataset_pairs(
    dataset: str, data_path: str
) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns (train_sources, test_sources, test_targets).
    Only test split is used for evaluation; train sources are kept for few-shot sampling.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(data_path)

    if dataset == "meqsum":
        if "CHQ" in df.columns:
            df = df.rename(columns={"CHQ": "question", "Summary": "target"})
        df = df[["question", "target"]].dropna()
        train_df, test_df = train_test_split(df, test_size=200, random_state=42)
    else:  # mqp
        q_cols = [c for c in df.columns if "question" in c.lower()]
        df = df.rename(columns={q_cols[0]: "question", q_cols[1]: "target"}) if len(q_cols) >= 2 else df
        df = df[["question", "target"]].dropna()
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    return (
        train_df["question"].tolist(),
        test_df["question"].tolist(),
        test_df["target"].tolist(),
    )


# ---------------------------------------------------------------------------
# Few-shot prompt builder
# ---------------------------------------------------------------------------

def build_few_shot_prefix(
    train_sources: List[str],
    train_targets: List[str],
    n_shots: int = 3,
    seed: int = 42,
) -> str:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train_sources), size=min(n_shots, len(train_sources)), replace=False)
    lines = ["Here are examples of consumer health question reformulation:\n"]
    for i in idx:
        lines.append(f"Input: {train_sources[i]}")
        lines.append(f"Reformulated: {train_targets[i]}\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str, adapter_dir: Optional[str] = None, hf_cache: Optional[str] = None):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    kwargs = {"cache_dir": hf_cache} if hf_cache else {}
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, **kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **kwargs)

    if adapter_dir and os.path.isdir(adapter_dir):
        model = PeftModel.from_pretrained(base, adapter_dir)
        print(f"Loaded LoRA adapter from {adapter_dir}")
    else:
        model = base

    model = model.to(device).eval()
    return tok, model, device


def zero_shot_reformulate(
    question: str,
    model,
    tokenizer,
    device: str,
    system: str,
    user: str,
    max_new_tokens: int = 80,
) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
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
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    sources: List[str],
    preds: List[str],
    refs: List[str],
    dataset_name: str = "",
    setting_name: str = "",
) -> Dict:
    import evaluate

    rouge_m = evaluate.load("rouge")
    bs_m    = evaluate.load("bertscore")
    bleu_m  = evaluate.load("bleu")

    rouge_out = rouge_m.compute(predictions=preds, references=refs, use_stemmer=True)
    bert_out  = bs_m.compute(predictions=preds, references=refs, lang="en")
    bleu_out  = bleu_m.compute(predictions=[p.split() for p in preds],
                                references=[[r.split()] for r in refs])

    mf1, chr_pct, per_cat = medfaith_f1(sources, preds, return_per_category=True)

    avg_fkgl = float(np.mean([fkgl(p) for p in preds]))

    results = {
        "setting":       setting_name,
        "dataset":       dataset_name,
        "rougeL":        round(rouge_out["rougeL"], 4),
        "bleu":          round(bleu_out["bleu"], 4),
        "bertscore_f1":  round(float(np.mean(bert_out["f1"])), 4),
        "Dx":            round(per_cat["Dx"], 4),
        "Rx":            round(per_cat["Rx"], 4),
        "Proc":          round(per_cat["Proc"], 4),
        "Fup":           round(per_cat["Fup"], 4),
        "medfaith_f1":   round(mf1, 4),
        "chr":           round(chr_pct, 2),
        "fkgl":          round(avg_fkgl, 1),
        "n_samples":     len(preds),
    }
    print_medfaith_report(mf1, chr_pct, per_cat, model_name=setting_name, dataset=dataset_name)
    return results


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def run_single_eval(
    sources: List[str],
    refs: List[str],
    model,
    tokenizer,
    device: str,
    setting: str,             # zero_shot | few_shot | peft
    retrieval: str,           # none | rag | ekg_rag
    ekg_pipeline: Optional[EKGRag],
    few_shot_prefix: str = "",
    max_new_tokens: int = 80,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Generate reformulations for all sources under the given (setting, retrieval) config.
    Returns (sources, predictions, references).
    """
    preds = []
    for i, q in enumerate(sources):
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(sources)}] generating...")

        if retrieval == "ekg_rag" and ekg_pipeline is not None:
            sys_p, usr_p = ekg_pipeline.build_prompt(q)
        elif retrieval == "rag" and ekg_pipeline is not None:
            # RAG-only: disable KG
            tmp = EKGRag(
                ekg_pipeline.retriever,
                umls_index=None,
                use_evidence=True,
                use_kg=False,
            )
            sys_p, usr_p = tmp.build_prompt(q)
        else:
            # Base: no retrieval
            sys_p = "You are a medical question reformulation specialist."
            usr_p = (
                "Convert the following consumer health question into a concise "
                "professional medical question. Preserve all diagnoses, medications, "
                "procedures, and follow-up intent.\n\n"
                f"Input Question: {q}\n\nReformulated Question:"
            )

        if setting == "few_shot" and few_shot_prefix:
            usr_p = few_shot_prefix + "\n\n" + usr_p

        pred = zero_shot_reformulate(q, model, tokenizer, device, sys_p, usr_p, max_new_tokens)
        preds.append(pred)

    return sources, preds, refs


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate EKG-RAG reformulation on MeQSum / MQP"
    )
    p.add_argument("--dataset",     choices=["meqsum", "mqp"], required=True)
    p.add_argument("--data_path",   type=str, required=True)
    p.add_argument("--model_name",  type=str, required=True,
                   help="HF checkpoint or path to merged model.")
    p.add_argument("--adapter_dir", type=str, default=None,
                   help="Path to LoRA adapter (for PEFT setting).")
    p.add_argument("--setting",
                   choices=["zero_shot", "few_shot", "peft"], required=True)
    p.add_argument("--retrieval",
                   choices=["none", "rag", "ekg_rag"], default="none")
    p.add_argument("--corpus_path", type=str, default=None,
                   help="One passage per line. Required for rag/ekg_rag.")
    p.add_argument("--umls_csv",    type=str, default=None,
                   help="UMLS triples CSV (head, relation, tail).")
    p.add_argument("--output_dir",  type=str, default="results/")
    p.add_argument("--hf_cache",    type=str, default=None)
    p.add_argument("--n_shots",     type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=80)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset : {args.dataset}")
    print(f"Model   : {args.model_name}")
    print(f"Setting : {args.setting}")
    print(f"Retrieval: {args.retrieval}")
    print(f"{'='*60}\n")

    # Load data
    train_srcs, test_srcs, test_refs = load_dataset_pairs(args.dataset, args.data_path)
    print(f"Test samples: {len(test_srcs)}")

    # Load model
    tok, model, device = load_model_and_tokenizer(
        args.model_name,
        adapter_dir=args.adapter_dir if args.setting == "peft" else None,
        hf_cache=args.hf_cache,
    )

    # EKG-RAG pipeline
    ekg_pipeline = None
    if args.retrieval in ("rag", "ekg_rag") and args.corpus_path:
        ekg_pipeline = build_ekg_rag_pipeline(
            corpus_path=args.corpus_path,
            umls_csv_path=args.umls_csv if args.retrieval == "ekg_rag" else None,
        )
        print("EKG-RAG pipeline loaded.")

    # Few-shot prefix
    few_shot_prefix = ""
    if args.setting == "few_shot":
        # We need train targets too — reload with targets
        import pandas as pd
        from sklearn.model_selection import train_test_split
        df = pd.read_csv(args.data_path)
        if "CHQ" in df.columns:
            df = df.rename(columns={"CHQ": "question", "Summary": "target"})
        df = df[["question", "target"]].dropna()
        n_test = 200 if args.dataset == "meqsum" else int(0.2 * len(df))
        train_df, _ = train_test_split(df, test_size=n_test, random_state=42)
        few_shot_prefix = build_few_shot_prefix(
            train_df["question"].tolist(), train_df["target"].tolist(), n_shots=args.n_shots
        )

    # Run inference
    srcs, preds, refs = run_single_eval(
        test_srcs, test_refs,
        model, tok, device,
        setting=args.setting,
        retrieval=args.retrieval,
        ekg_pipeline=ekg_pipeline,
        few_shot_prefix=few_shot_prefix,
        max_new_tokens=args.max_new_tokens,
    )

    # Compute metrics
    setting_label = f"{args.setting}_{args.retrieval}_{args.model_name.split('/')[-1]}"
    metrics = compute_all_metrics(srcs, preds, refs, args.dataset, setting_label)

    # Save outputs
    results_file = out_dir / f"{setting_label}_{args.dataset}.json"
    with open(results_file, "w") as f:
        json.dump({"metrics": metrics, "predictions": preds[:20]}, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Also save all predictions
    preds_file = out_dir / f"{setting_label}_{args.dataset}_preds.txt"
    with open(preds_file, "w") as f:
        for src, pred, ref in zip(srcs, preds, refs):
            f.write(f"SOURCE: {src}\nPRED:   {pred}\nREF:    {ref}\n\n")
    print(f"Predictions saved to: {preds_file}")


if __name__ == "__main__":
    main()
