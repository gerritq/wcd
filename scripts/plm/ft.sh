#!/bin/bash
#SBATCH --job-name=plm-xlm-b-no-context
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --exclude=erc-hpc-comp039,erc-hpc-comp054

nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_TYPE="vanilla" # vanilla
LANGUAGES=("en" "nl")
# LANGUAGES=("en")
MODEL_NAME="xlm-r-b"  # mDeberta-l mBert xlm-r-l xlm-r-b
SMOKE_TEST=0
CONTEXT=1
TRAINING_SIZE="-1"
NOTES="mBert full run no context"

# Total runs 12
# EPOCHS_LIST=(1) 
# LR_LIST=(5e-5)
# BATCH_LIST=(16)

# Total runs 12
EPOCHS_LIST=(1 2) 
LR_LIST=(5e-5 1e-5 5e-6)
BATCH_LIST=(16 32)

for LANG in "${LANGUAGES[@]}"; do
  for EPOCHS in "${EPOCHS_LIST[@]}"; do
    for LR in "${LR_LIST[@]}"; do
      for BS in "${BATCH_LIST[@]}"; do

        uv run ft.py \
          --model_type "$MODEL_TYPE" \
          --model_name "$MODEL_NAME" \
          --lang "$LANG" \
          --smoke_test "$SMOKE_TEST" \
          --context "$CONTEXT" \
          --training_size $TRAINING_SIZE \
          --notes "$NOTES" \
          --epochs "$EPOCHS" \
          --learning_rate "$LR" \
          --batch_size "$BS"

      done
    done
  done
done