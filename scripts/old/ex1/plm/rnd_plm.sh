#!/bin/bash
#SBATCH --job-name=rnd-plm
#SBATCH --output=../../../logs/%j.out
#SBATCH --error=../../../logs/%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# VARs
# "en" "nl" "no" "it" "pt" "ro" "ru" "uk" "bg" "id" "vi" "tr"
# LANGUAGES=("en" "nl" "no" "it" "pt" "ro")
LANGUAGES=("en" "nl" "no" "it" "pt" "ro" "ru" "uk" "bg" "id" "vi" "tr")
MODELS=("mDeberta-b") # "mBert" "mDeberta-b"
TRIALS=5
SMOKE_TEST=0

for LANGUAGE in "${LANGUAGES[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    echo "Running with LANGUAGE=$LANGUAGE, MODEL=$MODEL"
    start=$(date +%s)

    uv run rnd_plm.py \
      --lang "$LANGUAGE" \
      --model "$MODEL" \
      --trials $TRIALS \
      --smoke_test $SMOKE_TEST

    end=$(date +%s)
    runtime=$((end - start))
    echo "Time taken for LANGUAGE=$LANGUAGE, MODEL=$MODEL, TRIALS:$TRIALS: ${runtime}s"
  done
done
