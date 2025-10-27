#!/bin/bash
#SBATCH --job-name=ph-en
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=nmes_gpu,gpu,interruptible_gpu
# SBATCH --partition=nmes_gpu,gpu
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# VARs
# "en" "nl" "no" "it" "pt" "ro" "ru" "uk" "bg" "id" "vi" "tr"
LANGUAGES=("en")
MODELS=("mBert") # "mBert"
TRIALS=5

for LANGUAGE in "${LANGUAGES[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    echo "Running with LANGUAGE=$LANGUAGE, MODEL=$MODEL"
    start=$(date +%s)

    uv run plm_hp.py \
      --lang "$LANGUAGE" \
      --model "$MODEL" \
      --trials $TRIALS 

    end=$(date +%s)
    runtime=$((end - start))
    echo "Time taken for LANGUAGE=$LANGUAGE, MODEL=$MODEL, TRIALS:$TRIALS: ${runtime}s"
  done
done
