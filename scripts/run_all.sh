#!/bin/bash
#SBATCH --job-name=ekg_rag_meqsum
#SBATCH --account=project_2014607
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/project_2014607/logs/%x-%j.out
#SBATCH --error=/scratch/project_2014607/logs/%x-%j.err

set -euo pipefail

# --- Paths ---
BASE=/scratch/project_2014607
LOGS="$BASE/logs"
HF_HOME="$BASE/hf_cache"
MPLCONFIGDIR="$BASE/mpl_config"
NLTK_DATA="$BASE/nltk_data"
TMPDIR="$BASE/tmp"
PIP_CACHE_DIR="$BASE/pip_cache"
PYUSER="$BASE/pyuser"

DATA_DIR="$BASE/data"
MEQSUM_CSV="$DATA_DIR/meqsum.csv"
MQP_CSV="$DATA_DIR/mqp.csv"
CORPUS_PATH="$DATA_DIR/pubmed_medlineplus_passages.txt"
UMLS_CSV="$BASE/kg/umls_triples.csv"

mkdir -p "$LOGS" "$HF_HOME" "$MPLCONFIGDIR" "$NLTK_DATA" "$TMPDIR" \
         "$PIP_CACHE_DIR" "$PYUSER" "$DATA_DIR" "$BASE/kg" "$BASE/models" "$BASE/results"

cd "$BASE"

# --- Modules ---
module --force purge
module load pytorch

export HF_HOME
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export MPLCONFIGDIR
export NLTK_DATA
export TMPDIR
export PIP_CACHE_DIR
export PYTHONUSERBASE="$PYUSER"

pyver=$(python - <<'PY'
import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)
export PYTHONPATH="$PYUSER/lib/python${pyver}/site-packages:${PYTHONPATH:-}"
export PATH="$PYUSER/bin:$PATH"

# --- Install deps ---
python -m pip install --user --upgrade \
  "transformers>=4.44.0" \
  datasets peft accelerate \
  evaluate rouge-score bert-score \
  sentence-transformers \
  scikit-learn pandas \
  sacrebleu

# Optional: SPLADE
python -m pip install --user splade 2>/dev/null || echo "[warn] splade not available, dense fallback active"

python - <<'PY'
import torch, sys
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(), "| python", sys.version.split()[0])
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

# ====================================================================
# STEP 1: QLoRA fine-tuning on MeQSum (Qwen2.5-7B)
# ====================================================================
echo "[$(date)] Starting QLoRA fine-tuning on MeQSum..."

python models/train_qlora.py \
  --dataset meqsum \
  --data_path "$MEQSUM_CSV" \
  --checkpoint Qwen/Qwen2.5-7B-Instruct \
  --output_dir "$BASE/models/qwen_meqsum" \
  --epochs 3 \
  --lr 2e-4 \
  --batch_size 4 \
  --grad_accum 4 \
  --warmup_ratio 0.06 \
  --max_src_len 512 \
  --max_tgt_len 64 \
  --seed 42

QWEN_ADAPTER="$BASE/models/qwen_meqsum/adapter"

# ====================================================================
# STEP 2: Evaluate — all settings, MeQSum
# ====================================================================
echo "[$(date)] Evaluating Qwen2.5 PEFT + EKG-RAG on MeQSum..."

python evaluation/run_eval.py \
  --dataset meqsum \
  --data_path "$MEQSUM_CSV" \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --adapter_dir "$QWEN_ADAPTER" \
  --setting peft \
  --retrieval ekg_rag \
  --corpus_path "$CORPUS_PATH" \
  --umls_csv "$UMLS_CSV" \
  --output_dir "$BASE/results/" \
  --max_new_tokens 80

echo "[$(date)] Evaluating Qwen2.5 PEFT + RAG-only on MeQSum..."

python evaluation/run_eval.py \
  --dataset meqsum \
  --data_path "$MEQSUM_CSV" \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --adapter_dir "$QWEN_ADAPTER" \
  --setting peft \
  --retrieval rag \
  --corpus_path "$CORPUS_PATH" \
  --output_dir "$BASE/results/" \
  --max_new_tokens 80

echo "[$(date)] Evaluating Qwen2.5 PEFT base (no retrieval) on MeQSum..."

python evaluation/run_eval.py \
  --dataset meqsum \
  --data_path "$MEQSUM_CSV" \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --adapter_dir "$QWEN_ADAPTER" \
  --setting peft \
  --retrieval none \
  --output_dir "$BASE/results/" \
  --max_new_tokens 80

# Zero-shot baselines (no adapter)
echo "[$(date)] Zero-shot Qwen2.5 + EKG-RAG on MeQSum..."

python evaluation/run_eval.py \
  --dataset meqsum \
  --data_path "$MEQSUM_CSV" \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --setting zero_shot \
  --retrieval ekg_rag \
  --corpus_path "$CORPUS_PATH" \
  --umls_csv "$UMLS_CSV" \
  --output_dir "$BASE/results/" \
  --max_new_tokens 80

echo "[$(date)] Zero-shot Qwen2.5 base on MeQSum..."

python evaluation/run_eval.py \
  --dataset meqsum \
  --data_path "$MEQSUM_CSV" \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --setting zero_shot \
  --retrieval none \
  --output_dir "$BASE/results/" \
  --max_new_tokens 80

# ====================================================================
# STEP 3: MQP dataset
# ====================================================================
echo "[$(date)] Evaluating Qwen2.5 PEFT + EKG-RAG on MQP..."

python evaluation/run_eval.py \
  --dataset mqp \
  --data_path "$MQP_CSV" \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --adapter_dir "$QWEN_ADAPTER" \
  --setting peft \
  --retrieval ekg_rag \
  --corpus_path "$CORPUS_PATH" \
  --umls_csv "$UMLS_CSV" \
  --output_dir "$BASE/results/" \
  --max_new_tokens 80

echo "[$(date)] All jobs complete. Results in $BASE/results/"
