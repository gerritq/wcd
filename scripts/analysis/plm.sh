#!/bin/bash
#SBATCH --job-name=plm
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# "en" "nl" "no" "it" "pt" "ro" "ru" "uk" "bg" "id"
LANGUAGES=("nl" "no" "ro")
MODELS=("xlm-r" "mBert") # "xlm-r" "mBert"

for LANGUAGE in "${LANGUAGES[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    echo "Running with LANGUAGE=$LANGUAGE, MODEL=$MODEL"
    start=$(date +%s)

    uv run plm.py \
      --lang "$LANGUAGE" \
      --model "$MODEL"

    end=$(date +%s)
    runtime=$((end - start))
    echo "Time taken for LANGUAGE=$LANGUAGE, MODEL=$MODEL: ${runtime}s"
  done
done
