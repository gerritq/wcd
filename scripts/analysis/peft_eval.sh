#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# MODELS=("model_11")
# MODELS=($(seq -f "model_%g" 2 18))
MODELS=("model_4")
# MODELS=($(seq -f "model_%g" 19 31))
MODE="all" # all "ool"
LANGUAGES=("en" "ct_english")

for MODEL in "${MODELS[@]}"; do
  start=$(date +%s)

  uv run peft_eval.py \
    --languages "${LANGUAGES[@]}" \
    --model_number "$MODEL" \
    --mode "$MODE"

  end=$(date +%s)
  runtime=$((end - start))
  echo "Time taken for MODEL=$MODEL: ${runtime}s"
done