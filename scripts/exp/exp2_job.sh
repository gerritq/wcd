#!/bin/bash
#SBATCH --job-name=exp2
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --exclude=erc-hpc-comp054,erc-hpc-comp040
# comp050 slow
# comp039 has error

set -euo pipefail

nvidia-smi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# -------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------
TEST_LANG="$1"
RAW_TRAINING_LANGS="$2" # this is space separated!!
MODEL_TYPE="$3"
ATL="$4"
MODEL_NAME="$5"
TRAINING_SIZE="$6"

# this is to make an array from space sep. string
read -r -a TRAINING_LANGS <<< "$RAW_TRAINING_LANGS"

echo "Running with:"
echo "  TEST_LANG      = $TEST_LANG"
echo "  TRAINING_LANGS = ${TRAINING_LANGS[*]}"
echo "  MODEL_TYPE     = $MODEL_TYPE"
echo "  ATL            = $ATL"
echo "  MODEL_NAME     = $MODEL_NAME"
echo "  TRAINING_SIZE  = $TRAINING_SIZE"
echo

# -------------------------------------------------------------------
# EXP2 VARs
# -------------------------------------------------------------------
SMOKE_TEST=0
CONTEXT=1
EXP="cl"
QUANTIZATION=1
PROMPT_TEMPLATE="instruct"

# -------------------------------------------------------------------
# HP selection
# -------------------------------------------------------------------
if [ "$MODEL_TYPE" == "plm" ]; then
    # HPs PLMs
    EPOCHS=3
    LR_LIST=(5e-5)
    BATCH_SIZE_LIST=(32)
    GRAD_NORM_LIST=(1)
    WEIGHT_DECAY=0.01
else
    # HPs SLMs
    EPOCHS=3
    LR_LIST=(2e-4)
    BATCH_SIZE_LIST=(24)
    GRAD_NORM_LIST=(0.4)
    WEIGHT_DECAY=0.01
fi

# -------------------------------------------------------------------
# Run loop
# -------------------------------------------------------------------
for LR in "${LR_LIST[@]}"; do
  for BS in "${BATCH_SIZE_LIST[@]}"; do
    for GN in "${GRAD_NORM_LIST[@]}"; do

      uv run run.py \
        --model_type "$MODEL_TYPE" \
        --model_name "$MODEL_NAME" \
        --prompt_template "$PROMPT_TEMPLATE" \
        --quantization "$QUANTIZATION" \
        --context "$CONTEXT" \
        --smoke_test "$SMOKE_TEST" \
        --training_size "$TRAINING_SIZE" \
        --epochs "$EPOCHS" \
        --learning_rate "$LR" \
        --batch_size "$BS" \
        --max_grad_norm "$GN" \
        --weight_decay "$WEIGHT_DECAY" \
        --atl "$ATL" \
        --experiment "$EXP" \
        --training_langs "${TRAINING_LANGS[@]}" \
        --test_lang "$TEST_LANG"

    done
  done
done