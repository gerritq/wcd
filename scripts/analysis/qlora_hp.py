#!/bin/bash
#SBATCH --job-name=l3-old
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# VARs
LANGUAGES=("en")
MODELS=("llama3_8b")

# HP SEARCHSPACE
BATCH=(8 16)
EPOCHS=(1 3)
LEARNING_RATE=(1e-4)
LORA_R=(8, 16)
lora_alpha=(8, 16)
LEARNING_RATE=1e-4 # qlora paper

for LANGUAGE in "${LANGUAGES[@]}"; do
  for MODEL in "${MODELS[@]}"; do

    uv run qlora2.py \
      --lang "$LANGUAGE" \
      --model "$MODEL" \
      --learning_rate "$LEARNING_RATE" \
      --batch_size $BATCH \
      --epochs $EPOCHS \
      --plw $PLW
  done
done