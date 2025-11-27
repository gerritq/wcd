#!/bin/bash
#SBATCH --job-name=l8-atl-nl-lengths-context-expl
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --exclude=erc-hpc-comp050,erc-hpc-comp054,erc-hpc-comp039

# comp050 slow
# comp039 has error
nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# VARs
# Classifier not workign atm
MODEL_TYPE="atl" # vanilla atl classifier
LANG="nl"
MODEL_NAME="llama3_8b" # qwen3_06b llama3_8b qwen3_8b llama3_8b_base qwen3_8b_base
NOTES="-"

SMOKE_TEST=0
CONTEXT=1
TRAINING_SIZE_LIST=(500 1000 1500 2000 2500 3000 3500 4000 4500) # -1 := full training
# TRAINING_SIZE_LIST=(-1)
QUANTIZATION=1
SAVE_LAST_EPOCH=1

# HPs
EPOCHS=3
LR_LIST=(2e-4)
BATCH_LIST=(24)
GRAD_NORM_LIST=(0.4)
GRAD_ACCUMULATION_STEPS=1

# HPs (27 runs)
# GRAD_ACCUMULATION_STEPS=1
# EPOCHS_LIST=(3)
# LR_LIST=(5e-4 2e-4 5e-5)
# BATCH_LIST=(24)
# GRAD_NORM_LIST=(0.4 0.6 0.8)

for TRAINING_SIZE in "${TRAINING_SIZE_LIST[@]}"; do
  for LR in "${LR_LIST[@]}"; do
    for BS in "${BATCH_LIST[@]}"; do
      for GN in "${GRAD_NORM_LIST[@]}"; do

        uv run ft.py \
          --model_type "$MODEL_TYPE" \
          --model_name "$MODEL_NAME" \
          --lang "$LANG" \
          --quantization "$QUANTIZATION" \
          --context "$CONTEXT" \
          --smoke_test "$SMOKE_TEST" \
          --training_size "$TRAINING_SIZE" \
          --save_last_epoch "$SAVE_LAST_EPOCH" \
          --notes "$NOTES" \
          --epochs "$EPOCHS" \
          --learning_rate "$LR" \
          --batch_size "$BS" \
          --max_grad_norm "$GN" \
          --gradient_accumulation_steps "$GRAD_ACCUMULATION_STEPS"

      done
    done
  done
done