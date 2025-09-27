#!/bin/bash
#SBATCH --job-name=slm-ft
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=12:00:00
# SBATCH --partition=gpu,nmes_gpu,interruptible_gpu
#SBATCH --reservation=rental_8089
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# "llama_3_70b": "meta-llama/Meta-Llama-3-70B-Instruct",
# "llama_3_3b": "meta-llama/Meta-Llama-3-8B-Instruct",
# "qwen8b": "Qwen/Qwen3-8B",
# "qwen3b": "Qwen/Qwen3-30B-A3B",
# "qwen06b": "Qwen/Qwen3-0.6B",
# "qwen32b": "Qwen/Qwen3-32B"

DATASETS=("en_sents") #"cn_fa" "cn_fa_ss" "cn_fa_ss_nl"
MODELS=("qwen8b") # "qwen06b" "llama_3_7b"
N=10000
BATCH=4
EPOCHS=20

for DATA in "${DATASETS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    echo "Running with DATA=$DATA, MODEL=$MODEL"
    uv run ft.py \
      --data "$DATA" \
      --model "$MODEL" \
      --n "$N" \
      --batch_size $BATCH \
      --epochs $EPOCHS
  done
done
