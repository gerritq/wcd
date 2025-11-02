#!/bin/bash
#SBATCH --job-name=c-q4-all
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100

nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LANGUAGES=("it" "ro" "ru" "uk" "bg" "id" "vi" "tr")
MODELS=("qwen3_4b")

# qwen 8b euns 1:20 hrs for 2 epochs

BATCH=8
EPOCHS=(1 3)
GRAD_ACC=4 # unsloth says larger effective batch size leads to more stable training
MAX_GRAD_NORM=0.3 #  0.3 in the qlora paper for smaller models; 0.03 for bigger ones; defualts to 1 in the TrainerArguments class
LEARNING_RATES=(1e-4 1e-5) # qlora paper 2e-4

for LANGUAGE in "${LANGUAGES[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for LR in "${LEARNING_RATES[@]}"; do
      for EPOCH in "${EPOCHS[@]}"; do

        uv run classy.py \
          --lang "$LANGUAGE" \
          --model "$MODEL" \
          --learning_rate "$LR" \
          --batch_size "$BATCH" \
          --epochs "$EPOCH" \
          --grad_acc "$GRAD_ACC" \
          --max_grad_norm "$MAX_GRAD_NORM"

      done
    done
  done
done