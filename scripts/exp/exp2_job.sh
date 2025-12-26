#!/bin/bash
#SBATCH --job-name=exp3
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --exclude=erc-hpc-comp054,erc-hpc-comp050,erc-hpc-comp040,erc-hpc-comp038
# SBATCH --reservation=rental_8734

# comp050 slow
# comp039 has error

set -euo pipefail

nvidia-smi

export PYTORCH_ALLOC_CONF=expandable_segments:True

MODEL_TYPE="${MODEL_TYPE}"
MODEL_NAME="${MODEL_NAME}"
SOURCE_LANGS="${SOURCE_LANGS}"
TARGET_LANGS="${TARGET_LANGS}"
SEEDS="${SEEDS}"
LANG_SETTINGS="${LANG_SETTINGS}"
CL_SETTINGS="${CL_SETTINGS}"
LOWER_LR="${LOWER_LR}"

NEW_LEARNING_RATE="${NEW_LEARNING_RATE:-1e-5}"
LANG="${LANG:-en}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ATL="${ATL:-0}"
CONTEXT="${CONTEXT:-1}"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-instruct}"
TRAINING_SIZE="${TRAINING_SIZE:-5000}"
EXPERIMENT="${EXPERIMENT:-binary}"

echo "Running with:"
echo "  MODEL_TYPE = $MODEL_TYPE"
echo "  MODEL_NAME = $MODEL_NAME"
echo "  SOURCE_LANGS = $SOURCE_LANGS"
echo "  TARGET_LANGS = $TARGET_LANGS"
echo "  SEEDS = $SEEDS"
echo "  CONTEXT    = $CONTEXT"
echo "  ATL        = $ATL"
echo "  PROMPT_TEMPLATE = $PROMPT_TEMPLATE"
echo "  TRAINING_SIZE = $TRAINING_SIZE"
echo "  EXPERIMENT = $EXPERIMENT"
echo "  LOWER_LR = $LOWER_LR"
echo "  NEW_LEARNING_RATE = $NEW_LEARNING_RATE"
echo "  BATCH_SIZE = $BATCH_SIZE"
echo

# VARS
SMOKE_TEST=0

# --------------------------------------------------------------------------------------------------
# HP SELECTION
# --------------------------------------------------------------------------------------------------

# Default will be overwritten
EPOCHS=3
LR=2e-4
BATCH_SIZE="$BATCH_SIZE"
GRAD_NORM=1
WEIGHT_DECAY=0.01

# --------------------------------------------------------------------------------------------------
# HP LOOP
# --------------------------------------------------------------------------------------------------


uv run cl.py \
  --experiment "$EXPERIMENT" \
  --model_type "$MODEL_TYPE" \
  --model_name "$MODEL_NAME" \
  --prompt_template "$PROMPT_TEMPLATE" \
  --lang "$LANG" \
  --context "$CONTEXT" \
  --atl "$ATL" \
  --smoke_test "$SMOKE_TEST" \
  --training_size "$TRAINING_SIZE" \
  --epochs "$EPOCHS" \
  --learning_rate "$LR" \
  --batch_size "$BATCH_SIZE" \
  --max_grad_norm "$GRAD_NORM" \
  --weight_decay "$WEIGHT_DECAY" \
  --seeds "$SEEDS" \
  --source_langs "$SOURCE_LANGS" \
  --target_langs "$TARGET_LANGS" \
  --lang_settings "$LANG_SETTINGS" \
  --cl_settings "$CL_SETTINGS" \
  --lower_lr "$LOWER_LR" \
  --new_learning_rate "$NEW_LEARNING_RATE"
