#!/bin/bash
#SBATCH --job-name=l8-m
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# "llama_3_70b": "meta-llama/Meta-Llama-3-70B-Instruct",
# "llama_3_8b": "meta-llama/Meta-Llama-3-8B-Instruct",
# "qwen8b": "Qwen/Qwen3-8B",
# "qwen3b": "Qwen/Qwen3-30B-A3B",
# "qwen06b": "Qwen/Qwen3-0.6B",
# "qwen32b": "Qwen/Qwen3-32B"

LANGUAGES=("ct_english")
MODELS=(
  "llama3_1b"
)
BATCH=8
EPOCHS=5

for LANGUAGE in "${LANGUAGES[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    echo "Running with DATA=$DATA, MODEL=$MODEL"
    uv run peft_run.py \
      --lang "$LANGUAGE" \
      --model "$MODEL" \
      --batch_size $BATCH \
      --epochs $EPOCHS
  done
done
