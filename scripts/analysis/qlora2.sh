#!/bin/bash
#SBATCH --job-name=q-l8-
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=05:30:00
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# "llama3_1b": "meta-llama/Llama-3.2-1B-Instruct",
# "llama3_3b": "meta-llama/Llama-3.2-3B-Instruct",
# "llama3_8b": "meta-llama/Llama-3.1-8B-Instruct",
# "llama3_70b": "meta-llama/Llama-3.3-70B-Instruct",
# "qwen3_06b": "Qwen/Qwen3-0.6B",
# "qwen3_4b": "Qwen/Qwen3-4B-Instruct-2507",
# "qwen3_8b": "Qwen/Qwen3-8B",
# "qwen3_30b": "Qwen/Qwen3-30B-A3B-Instruct-2507",
# "qwen3_32b": "Qwen/Qwen3-32B",
# "gemma3_12b": "google/gemma-3-12b-it",


# LANGUAGES=("en" "nl" "no" "it" "pt" "ro")
# LANGUAGES=("ru" "uk" "bg" "id" "vi" "tr")
# LANGUAGES=("en" "nl" "no" "it" "pt" "ro")
LANGUAGES=("en" "nl")

MODELS=("llama3_8b")

# "qwen3_06b"
# "qwen3_8b"
# "llama3_8b"

BATCH=8
EPOCHS=3
LEARNING_RATE=1e-4 # qlora paper 2e-4
GRAD_ACC=4
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=0.3
WARMUP_RATIO=0.03

for LANGUAGE in "${LANGUAGES[@]}"; do
  for MODEL in "${MODELS[@]}"; do

    uv run qlora2.py \
      --lang "$LANGUAGE" \
      --model "$MODEL" \
      --learning_rate "$LEARNING_RATE" \
      --batch_size $BATCH \
      --epochs $EPOCHS \
      --grad_acc "$GRAD_ACC" \
      --warmup_ratio "$WARMUP_RATIO" \
      --weight_decay "$WEIGHT_DECAY" \
      --max_grad_norm "$MAX_GRAD_NORM"
  done
done