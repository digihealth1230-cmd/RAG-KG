"""
models/train_qlora.py

QLoRA fine-tuning for LLaMA-3-8B-Instruct and Qwen2.5-7B-Instruct
on MeQSum and MQP for medical question reformulation.

Configuration follows Table 1:
  rank r=8, alpha=16, dropout=0.05
  target modules: query and value projections
  epochs=3, batch=16, lr=2e-4 cosine with 6% warmup
"""

import os
import sys
import time
import json
import random
import pathlib
import argparse
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

# Silence verbose HF warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    GenerationConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import evaluate

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from medfaith_f1 import medfaith_f1


# ---------------------------------------------------------------------------
# Prompt template (mirrors ekg_rag.py for consistency)
# ---------------------------------------------------------------------------

_SYSTEM = "You are a medical question reformulation specialist."

_USER_TMPL = (
    "Convert the following consumer health question into a concise professional "
    "medical question. Preserve all diagnoses, medications, procedures, and "
    "follow-up intent. Do not add unsupported information.\n\n"
    "Evidence Context:\n«BEGIN»\n{evidence}\n«END»\n\n"
    "Knowledge Context:\n{knowledge}\n\n"
    "Input Question: {question}\n\n"
    "Reformulated Question:"
)


def _make_user_prompt(
    question: str,
    evidence: str = "(no evidence)",
    knowledge: str = "(no knowledge context)",
) -> str:
    return _USER_TMPL.format(
        evidence=evidence, knowledge=knowledge, question=question
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_meqsum(split_path: str) -> Tuple[Dataset, Dataset]:
    """
    Load MeQSum from a two-column CSV (CHQ, Summary).
    Returns (train_dataset, test_dataset) using 800/200 split.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(split_path)
    if "CHQ" in df.columns:
        df = df.rename(columns={"CHQ": "question", "Summary": "target"})
    df = df[["question", "target"]].dropna()

    # Fixed 800/200 split as in the paper
    n_test = min(200, int(0.2 * len(df)))
    train_df, test_df = train_test_split(df, test_size=n_test, random_state=42)
    return Dataset.from_pandas(train_df.reset_index(drop=True)), \
           Dataset.from_pandas(test_df.reset_index(drop=True))


def load_mqp(split_path: str) -> Tuple[Dataset, Dataset]:
    """
    Load Medical Question Pair dataset (question1, question2 columns).
    We treat question1 as source and question2 as the reference reformulation.
    Returns (train_dataset, test_dataset) using 80/20 split.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(split_path)
    # Accept various column name conventions
    q_cols = [c for c in df.columns if "question" in c.lower() or "query" in c.lower()]
    if len(q_cols) >= 2:
        df = df.rename(columns={q_cols[0]: "question", q_cols[1]: "target"})
    df = df[["question", "target"]].dropna()

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    return Dataset.from_pandas(train_df.reset_index(drop=True)), \
           Dataset.from_pandas(test_df.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def make_tokenise_fn(tokenizer, max_src_len: int, max_tgt_len: int):
    """
    Returns a map function that converts (question, target) rows into
    (input_ids, attention_mask, labels) for causal LM training.
    Prompt tokens are masked with -100 so only the target is supervised.
    """
    def _fn(batch):
        convs = []
        for q in batch["question"]:
            convs.append([
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _make_user_prompt(q)},
            ])

        prompt_texts = tokenizer.apply_chat_template(
            convs, add_generation_prompt=True, tokenize=False
        )
        targets = [str(t) for t in batch["target"]]
        full_texts = [p + t for p, t in zip(prompt_texts, targets)]

        prompt_enc = tokenizer(
            prompt_texts,
            truncation=True, max_length=max_src_len, padding="max_length",
        )
        full_enc = tokenizer(
            full_texts,
            truncation=True, max_length=max_src_len + max_tgt_len, padding="max_length",
        )

        labels = []
        for full_ids, p_mask in zip(full_enc["input_ids"], prompt_enc["attention_mask"]):
            prompt_len = int(sum(p_mask))
            lab = list(full_ids)
            for i in range(prompt_len):
                lab[i] = -100
            labels.append(lab)

        return {
            "input_ids": full_enc["input_ids"],
            "attention_mask": full_enc["attention_mask"],
            "labels": labels,
        }

    return _fn


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def build_model_and_tokenizer(checkpoint: str, hf_cache: Optional[str] = None):
    """
    Load base model and tokenizer, then wrap with QLoRA.
    LoRA config matches Table 1: r=8, alpha=16, dropout=0.05, q+v targets.
    """
    kwargs = {}
    if hf_cache:
        kwargs["cache_dir"] = hf_cache

    tok = AutoTokenizer.from_pretrained(checkpoint, use_fast=True, **kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=dtype, **kwargs
    )
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    base.config.pad_token_id = tok.pad_token_id

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(base, lora_cfg)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"LoRA trainable params: {n_trainable:,} / {n_total:,} "
          f"({100 * n_trainable / n_total:.3f}%)")

    return tok, model


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def decode_batch(tokenizer, preds, labels):
    pred_ids = np.where(preds != -100, preds, tokenizer.pad_token_id)
    lab_ids  = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(lab_ids,   skip_special_tokens=True)
    return (
        [p.strip() for p in decoded_preds],
        [l.strip() for l in decoded_labels],
    )


def build_compute_metrics(tokenizer, source_questions: List[str]):
    rouge_metric = evaluate.load("rouge")
    bertscore_metric = evaluate.load("bertscore")

    # Capture source questions from outer scope for MedFaith-F1
    _sources = source_questions[:]

    def _metrics(eval_pred):
        preds, labels = eval_pred.predictions, eval_pred.label_ids
        decoded_preds, decoded_labels = decode_batch(tokenizer, preds, labels)

        rouge_out = rouge_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        bert_out = bertscore_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            lang="en",
        )

        # MedFaith-F1 on the first len(decoded_preds) sources
        n = len(decoded_preds)
        sources_slice = _sources[:n]
        mf1, chr_pct, per_cat = medfaith_f1(
            sources_slice, decoded_preds, return_per_category=True
        )

        return {
            "rougeL":          rouge_out["rougeL"],
            "bleu":            rouge_out.get("rouge1", 0.0),  # proxy; use sacrebleu if needed
            "bertscore_f1":    float(np.mean(bert_out["f1"])),
            "medfaith_f1":     mf1,
            "chr":             chr_pct,
            "f1_dx":           per_cat["Dx"],
            "f1_rx":           per_cat["Rx"],
            "f1_proc":         per_cat["Proc"],
            "f1_fup":          per_cat["Fup"],
        }

    return _metrics


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="QLoRA fine-tuning for medical query reformulation (EKG-RAG paper)"
    )
    p.add_argument("--dataset",     choices=["meqsum", "mqp"], default="meqsum")
    p.add_argument("--data_path",   type=str, required=True,
                   help="Path to dataset CSV.")
    p.add_argument("--checkpoint",  type=str,
                   default="Qwen/Qwen2.5-7B-Instruct",
                   help="HuggingFace checkpoint. Use meta-llama/Meta-Llama-3-8B-Instruct for LLaMA.")
    p.add_argument("--output_dir",  type=str, default="models/qlora_out")
    p.add_argument("--hf_cache",    type=str, default=None)

    # Training hyperparams (Table 1)
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--batch_size",  type=int,   default=4,
                   help="Per-device batch size. Effective batch=16 via grad accum.")
    p.add_argument("--grad_accum",  type=int,   default=4)
    p.add_argument("--warmup_ratio",type=float, default=0.06)
    p.add_argument("--max_src_len", type=int,   default=512)
    p.add_argument("--max_tgt_len", type=int,   default=64)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--eval_steps",  type=int,   default=100)
    p.add_argument("--save_merged", action="store_true",
                   help="Merge LoRA weights into base and save full model.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    stamp = time.strftime("%Y%m%d-%H%M")
    out_root = pathlib.Path(args.output_dir) / f"{args.dataset}_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"torch {torch.__version__} | cuda: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")

    # Load data
    loader = load_meqsum if args.dataset == "meqsum" else load_mqp
    train_ds, test_ds = loader(args.data_path)
    print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")

    source_questions = list(test_ds["question"])

    # Model
    tok, model = build_model_and_tokenizer(args.checkpoint, args.hf_cache)

    # Tokenise
    tok_fn = make_tokenise_fn(tok, args.max_src_len, args.max_tgt_len)
    cols = train_ds.column_names
    tok_train = train_ds.map(tok_fn, batched=True, remove_columns=cols)
    tok_test  = test_ds.map(tok_fn,  batched=True, remove_columns=cols)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tok, model=model, padding="longest", label_pad_token_id=-100
    )

    training_args = TrainingArguments(
        output_dir=str(out_root),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="medfaith_f1",
        greater_is_better=True,
        save_total_limit=2,
        remove_unused_columns=False,
        predict_with_generate=True,
        generation_max_length=args.max_tgt_len,
        report_to="none",
        seed=args.seed,
    )

    compute_metrics = build_compute_metrics(tok, source_questions)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_train,
        eval_dataset=tok_test,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # Save adapter
    adapter_dir = out_root / "adapter"
    tok.save_pretrained(adapter_dir)
    trainer.model.save_pretrained(adapter_dir)
    print(f"LoRA adapter saved to: {adapter_dir}")

    # Optionally merge weights
    if args.save_merged:
        merged_dir = out_root / "merged"
        merged_dir.mkdir(exist_ok=True)
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tok.save_pretrained(merged_dir)
        print(f"Merged model saved to: {merged_dir}")

    # Final eval
    final_metrics = trainer.evaluate()
    metrics_path = out_root / "metrics_final.json"
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Final metrics: {json.dumps(final_metrics, indent=2)}")
    print(f"Saved to: {metrics_path}")


if __name__ == "__main__":
    main()
