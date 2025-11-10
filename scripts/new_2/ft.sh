#!/bin/bash
#SBATCH --job-name=l8-c-nl-run-lr-5-epoch3
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --exclude=erc-hpc-comp050,erc-hpc-comp054

nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# VARs
MODEL_TYPE="atl" # vanilla atl classifier
LANG="nl"
MODEL_NAME="llama3_8b" # qwen3_06b llama3_8b qwen3_8b
SMOKE_TEST=0
QUANTIZATION=1
NOTES="vary batch and grad norm"
EXPLANATION=0
TRAINING_SIZE=-1 # -1 := full training

# HPs
GRAD_ACCUMULATION_STEPS=1
EPOCHS_LIST=(3) # 2 3 4
LR_LIST=(5e-4) # 5e-4 1e-4 5e-5
BATCH_LIST=(24 32) # og 8 with grad acc 4
GRAD_NORM_LIST=(0.4 0.6 0.8) # 0.2 0.4 0.6


# HPs
# EPOCHS_LIST=(1 2 3) # 2 3 4
# LR_LIST=(5e-4 1e-4 5e-5) # 5e-4 1e-4 5e-5
# BATCH_LIST=(16 32) # og 8 with grad acc 4
# GRAD_NORM_LIST=(0.2 0.4 0.6) # 0.2 0.4 0.6


for EPOCHS in "${EPOCHS_LIST[@]}"; do
  for LR in "${LR_LIST[@]}"; do
    for BS in "${BATCH_LIST[@]}"; do
      for GN in "${GRAD_NORM_LIST[@]}"; do
        
        uv run ft.py \
          --model_type "$MODEL_TYPE" \
          --model_name "$MODEL_NAME" \
          --lang "$LANG" \
          --quantization "$QUANTIZATION" \
          --explanation "$EXPLANATION" \
          --smoke_test "$SMOKE_TEST" \
          --training_size $TRAINING_SIZE \
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
