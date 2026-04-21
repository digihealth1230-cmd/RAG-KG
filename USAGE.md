# Usage Guide

This document provides detailed instructions for reproducing the results from the paper.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

For SPLADE sparse retrieval (optional, falls back to dense-only if unavailable):
```bash
pip install splade
```

### 2. Prepare Data

Download datasets following `data/README.md`, then place:
- `data/meqsum.csv` — MeQSum dataset (CHQ, Summary columns)
- `data/mqp.csv` — Medical Question Pair dataset (question1, question2)
- `data/pubmed_medlineplus_passages.txt` — Evidence corpus (one passage per line)
- `kg/umls_triples.csv` — UMLS knowledge graph (head, relation, tail columns)

### 3. Run Demo

```bash
python demo.py
```

This demonstrates the full EKG-RAG pipeline on a single example without requiring downloaded data.

---

## Reproducing Paper Results

### Table 2: Main Results

To reproduce the full Table 2 results (all settings × models × datasets):

**Step 1: Fine-tune with QLoRA**

```bash
# Qwen2.5-7B on MeQSum
python models/train_qlora.py \
  --dataset meqsum \
  --data_path data/meqsum.csv \
  --checkpoint Qwen/Qwen2.5-7B-Instruct \
  --output_dir models/qwen_meqsum \
  --epochs 3 --lr 2e-4 --batch_size 4 --grad_accum 4

# LLaMA-3-8B on MeQSum
python models/train_qlora.py \
  --dataset meqsum \
  --data_path data/meqsum.csv \
  --checkpoint meta-llama/Meta-Llama-3-8B-Instruct \
  --output_dir models/llama_meqsum \
  --epochs 3 --lr 2e-4 --batch_size 4 --grad_accum 4
```

**Step 2: Evaluate across all configurations**

For each (model, setting, retrieval, dataset) combination:

```bash
# Example: Qwen PEFT + EKG-RAG on MeQSum (best result: MedFaith-F1=0.7325, CHR=26.75%)
python evaluation/run_eval.py \
  --dataset meqsum \
  --data_path data/meqsum.csv \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --adapter_dir models/qwen_meqsum/adapter \
  --setting peft \
  --retrieval ekg_rag \
  --corpus_path data/pubmed_medlineplus_passages.txt \
  --umls_csv kg/umls_triples.csv \
  --output_dir results/

# Zero-shot baseline
python evaluation/run_eval.py \
  --dataset meqsum \
  --data_path data/meqsum.csv \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --setting zero_shot \
  --retrieval none \
  --output_dir results/

# Few-shot + RAG
python evaluation/run_eval.py \
  --dataset meqsum \
  --data_path data/meqsum.csv \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --setting few_shot \
  --retrieval rag \
  --corpus_path data/pubmed_medlineplus_passages.txt \
  --n_shots 3 \
  --output_dir results/
```

Results are saved as JSON files in `results/`.

---

## HPC Cluster Execution (SLURM)

For batch processing on an HPC cluster with SLURM:

```bash
sbatch scripts/run_all.sh
```

This script:
1. Fine-tunes Qwen2.5-7B on MeQSum
2. Evaluates across all settings (zero-shot, few-shot, PEFT) × retrieval configs (none, RAG, EKG-RAG)
3. Saves all results to `/scratch/project_2014607/results/`

Modify paths in `scripts/run_all.sh` to match your cluster setup.

---

## Evaluating Custom Models

To evaluate a custom or external model:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from ekg_rag import build_ekg_rag_pipeline, make_hf_model_fn

# Load your model
tokenizer = AutoTokenizer.from_pretrained("path/to/model")
model = AutoModelForCausalLM.from_pretrained("path/to/model").to("cuda")

# Build pipeline
pipeline = build_ekg_rag_pipeline(
    corpus_path="data/pubmed_medlineplus_passages.txt",
    umls_csv_path="kg/umls_triples.csv",
)

# Wrap model
model_fn = make_hf_model_fn(model, tokenizer, max_new_tokens=80)

# Reformulate
question = "Your consumer health question here"
reformulation = pipeline.reformulate(question, model_fn)
print(reformulation)
```

---

## Metric-Only Evaluation

If you have existing predictions and want to compute MedFaith-F1:

```python
from medfaith_f1 import medfaith_f1, print_medfaith_report

sources = ["source question 1", "source question 2", ...]
preds = ["predicted reformulation 1", "predicted reformulation 2", ...]

mf1, chr, per_cat = medfaith_f1(sources, preds, return_per_category=True)
print_medfaith_report(mf1, chr, per_cat, model_name="YourModel", dataset="MeQSum")
```

---

## Configuration

All hyperparameters are in `configs/default.yaml`. To modify:

```yaml
# Example: increase LoRA rank
qlora:
  rank: 16  # default: 8
  alpha: 32 # default: 16

# Example: increase retrieval budget
retrieval:
  token_budget: 768  # default: 512
```

Load in code:
```python
import yaml
with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)
```

---

## Troubleshooting

**Issue**: `ImportError: No module named 'splade'`
- **Solution**: SPLADE is optional. The retriever will automatically fall back to dense-only (SapBERT) retrieval if SPLADE is unavailable.

**Issue**: Out of memory during QLoRA training
- **Solution**: Reduce `batch_size` and increase `grad_accum` to maintain the same effective batch size (16):
  ```bash
  --batch_size 2 --grad_accum 8
  ```

**Issue**: UMLS concept extraction returns empty
- **Solution**: This is expected if no UMLS CSV is provided. The pipeline will still work with retrieval-only (RAG). To enable full EKG-RAG, provide a valid UMLS triples CSV.

**Issue**: "HuggingFace model X not found"
- **Solution**: Some models require authentication. Set your HF token:
  ```bash
  huggingface-cli login
  ```

---

## Citation

If you use this code, please cite our paper:

```
Anonymous submission. Citation information will be added upon acceptance.
```
