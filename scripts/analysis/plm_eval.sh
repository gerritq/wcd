#!/bin/bash
#SBATCH --job-name=plm-hp
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=nmes_gpu,gpu,interruptible_gpu
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODELS=("model_26")
# MODELS=($(seq -f "model_%g" 1 21))
MODE="all" # all "ool"
LANGUAGES=("en" "nl" "no" "it" "pt" "ro" "ru" "uk" "bg" "id")

for MODEL in "${MODELS[@]}"; do
  start=$(date +%s)

  uv run plm_eval.py \
    --languages "${LANGUAGES[@]}" \
    --model "$MODEL" \
    --mode "$MODE"

  end=$(date +%s)
  runtime=$((end - start))
  echo "Time taken for MODEL=$MODEL: ${runtime}s"
done