#!/bin/bash
#SBATCH --job-name=slm-nE
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu,nmes_gpu,interruptible_gpu
#SBATCH --mem=20GB
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100_40g

nvidia-smi

# "llama_3_70b": "meta-llama/Meta-Llama-3-70B-Instruct",
# "llama_3_3b": "meta-llama/Meta-Llama-3-8B-Instruct",
# "qwen8b": "Qwen/Qwen3-8B",
# "qwen3b": "Qwen/Qwen3-30B-A3B",
# "qwen06b": "Qwen/Qwen3-0.6B",
# "qwen32b": "Qwen/Qwen3-32B"

DATASETS=("pt_sents" "pl_sents" "pu_sents") #"cn_fa" "cn_fa_ss" "cn_fa_ss_nl"
# MODELS=("llama_3_8b") "llama_3_70b"
MODELS=(
  "llama_3_8b"
  "qwen8b"
  "qwen3b"
  "qwen06b"
  "qwen32b"
)
N=500
BATCH=8

for DATA in "${DATASETS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    echo "Running with DATA=$DATA, MODEL=$MODEL"
    uv run slm.py \
      --data "$DATA" \
      --model "$MODEL" \
      --n "$N" \
      --batch_size $BATCH
  done
done
