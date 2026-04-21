# Beyond Lexical Similarity: Evaluating Faithfulness in LLM-Based Medical Question Reformulation



We introduce **MedFaith-F1**, a category-level faithfulness metric for medical query rewriting, and **EKG-RAG**, an evidence and knowledge-grounded reformulation framework. Together they expose a systematic gap between surface-level similarity metrics (ROUGE, BERTScore) and actual clinical information retention.

---

## Overview

Consumer health questions submitted to retrieval systems are verbose and informal. LLMs can compress them into professional queries, but standard metrics do not check whether diagnoses, medications, procedures, or follow-up intent survive that compression. This repo provides:

- `ekg_rag.py` — end-to-end reformulation pipeline (sparse-dense retrieval + UMLS KG grounding)
- `medfaith_f1.py` — category-level faithfulness metric (Dx / Rx / Proc / Fup)
- `sapbert_extractor.py` — SapBERT-based clinical concept extraction and UMLS normalization
- `retrieval/splade_retriever.py` — SPLADE-v2 sparse retrieval with MMR diversification
- `retrieval/dense_reranker.py` — SapBERT dense re-ranking for evidence selection
- `models/train_qlora.py` — QLoRA fine-tuning for LLaMA-3 and Qwen2.5
- `evaluation/run_eval.py` — full evaluation loop across zero-shot, few-shot, and PEFT settings
- `data/meqsum_loader.py` — MeQSum dataset loading and preprocessing
- `data/mqp_loader.py` — Medical Question Pair dataset loading

---

## Results

Best configuration: Qwen2.5-7B + EKG-RAG + QLoRA on MeQSum

| Metric | Value |
|---|---|
| ROUGE-L | 0.54 |
| BERTScore | 0.87 |
| MedFaith-F1 | 0.7325 |
| CHR (↓) | 26.75% |
| FKGL | 9.3 |

Category-level hallucination rates in zero-shot base models exceed 40%, invisible to ROUGE/BERTScore.

---

## Requirements

```bash
pip install torch transformers>=4.44.0 datasets peft accelerate
pip install evaluate rouge-score bert-score
pip install sentence-transformers
pip install splade
pip install requests biopython
```

For UMLS concept normalization, a UMLS API key is required (free registration at https://uts.nlm.nih.gov/).

---

## Quick Start

### Zero-shot evaluation
```bash
python evaluation/run_eval.py \
  --dataset meqsum \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --setting zero_shot \
  --retrieval none \
  --output_dir results/
```

### EKG-RAG + QLoRA (full pipeline)
```bash
# Step 1: Fine-tune
python models/train_qlora.py \
  --dataset meqsum \
  --checkpoint Qwen/Qwen2.5-7B-Instruct \
  --output_dir models/qwen_meqsum_qlora \
  --epochs 3 --lr 2e-4 --rank 8

# Step 2: Evaluate with EKG-RAG
python evaluation/run_eval.py \
  --dataset meqsum \
  --model_name models/qwen_meqsum_qlora/adapter \
  --setting peft \
  --retrieval ekg_rag \
  --output_dir results/
```

---

## Data

**MeQSum** (Abacha & Demner-Fushman, 2019): Consumer health questions from the U.S. National Library of Medicine. We use an 800/200 train/test split.

**Medical Question Pair (MQP)** (Wang et al., 2022): Paired medical questions with semantic similarity annotations. We use an 80/20 split.

Neither dataset requires special licensing for research use. See `data/` for loading scripts.

---

## Metric Details

### MedFaith-F1

For a source question `q` and its reformulation `q_hat`, we extract UMLS-normalized concepts in each of four categories C = {Dx, Rx, Proc, Fup} using SapBERT. A category is labelled *present* if at least one concept is identified. Per-category F1 scores are macro-averaged:

```
MedFaith-F1 = (1/4) * sum_c F1_c
CHR = (1 - MedFaith-F1) * 100
```

See `medfaith_f1.py` for the full implementation.

### EKG-RAG

Two parallel grounding channels:
1. **Evidence (E)**: SPLADE-v2 sparse retrieval over PubMed + MedlinePlus → SapBERT dense re-ranking → budgeted MMR diversification
2. **Knowledge (K)**: SapBERT entity linking → UMLS concept normalization (SNOMED for Dx/Proc, RxNorm for Rx) → ontology-guided subgraph construction

---

## Citation

Anonymous submission. Citation information will be added upon acceptance.

---

## License

Code released under MIT License. Dataset usage follows original dataset licenses.
