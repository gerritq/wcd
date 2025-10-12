#!/bin/bash
#SBATCH --job-name=plm-hp
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODELS=("model_1")
LANGUAGES=("en")

for MODEL in "${MODELS[@]}"; do
  echo "Running MODEL=$MODEL with LANGUAGES=${LANGUAGES[*]}"
  start=$(date +%s)

  uv run plm.py \
    --languages "${LANGUAGES[@]}" \
    --model "$MODEL"

  end=$(date +%s)
  runtime=$((end - start))
  echo "Time taken for MODEL=$MODEL: ${runtime}s"
done