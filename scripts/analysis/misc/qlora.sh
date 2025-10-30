#!/bin/bash
#SBATCH --job-name=l8-bg
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# TIME: llama8b takes about 45 mins

# VARs
#LANGUAGES=("en" "nl" "no" "it" "pt" "ro" "ru" "uk" "bg" "id" "vi" "tr")
LANGUAGES=("en")
MODELS=("qwen3_06b")

# HPs
LEARNING_RATE=1e-4
BATCH=1
EPOCHS=5
PLW=0

for LANGUAGE in "${LANGUAGES[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    echo "Running with MODEL=$MODEL, LANGUAGES=$LANGUAGE"
    uv run qlora.py \
      --lang "$LANGUAGE" \
      --model "$MODEL" \
      --learning_rate "$LEARNING_RATE" \
      --batch_size $BATCH \
      --epochs $EPOCHS
  done
done