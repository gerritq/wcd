#!/bin/bash
#SBATCH --job-name=mbart_ft_id
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --constraint=b200|a100|b200
# SBATCH --exclude=erc-hpc-comp054,erc-hpc-comp050,erc-hpc-comp040,erc-hpc-comp038
#SBATCH --exclude=erc-hpc-comp054

set -e

# to run: set smoke test to 0 and comment out the sinlge hp set

nvidia-smi
export PYTORCH_ALLOC_CONF=expandable_segments:True

LANG="id"
MODEL_NAME="facebook/mbart-large-50"
SEED="42"
SMOKE_TEST=0
USE_QLORA=1

WEIGHT_DECAY="0.0"
GRAD_ACCUM="1"
MAX_GRAD_NORM="1.0"
WARMUP_RATIO="0.03"
MAX_SOURCE_LENGTH="512"
MAX_NEW_TOKENS="8"
EVAL_STEPS="100"

# Hyperparameter grid.
BATCH_SIZE_LIST=(16)
LEARNING_RATE_LIST=(2e-4)
EPOCHS_LIST=(3)

# BATCH_SIZE_LIST=(8 16)
# LEARNING_RATE_LIST=(2e-4 5e-5 5e-6)
# EPOCHS_LIST=(1 2 3)

RUN_IDX=0
for BS in "${BATCH_SIZE_LIST[@]}"; do
  for LR in "${LEARNING_RATE_LIST[@]}"; do
    for EPOCHS in "${EPOCHS_LIST[@]}"; do

      uv run new_models.py \
        --lang "${LANG}" \
        --model_name "${MODEL_NAME}" \
        --seed "${SEED}" \
        --epochs "${EPOCHS}" \
        --learning_rate "${LR}" \
        --batch_size "${BS}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --max_grad_norm "${MAX_GRAD_NORM}" \
        --warmup_ratio "${WARMUP_RATIO}" \
        --max_source_length "${MAX_SOURCE_LENGTH}" \
        --eval_steps "${EVAL_STEPS}" \
        --smoke_test "${SMOKE_TEST}" \
        --use_qlora "${USE_QLORA}" \
        --run_idx "${RUN_IDX}"
    
    RUN_IDX=$((RUN_IDX + 1))
    
    done
  done
done
