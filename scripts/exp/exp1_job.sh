#!/bin/bash
#SBATCH --job-name=exp1
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=nmes_gpu,gpu,interruptible_gpu
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
# SBATCH --constraint=a100
#SBATCH --exclude=erc-hpc-comp054,erc-hpc-comp034,erc-hpc-comp040,erc-hpc-comp052,erc-hpc-comp050,erc-hpc-comp048,erc-hpc-comp049

# comp050 slow
# comp039 has error

set -euo pipefail

nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


LANG="${LANG}"
CONTEXT="${CONTEXT}"
MODEL_TYPE="${MODEL_TYPE}"
MODEL_NAME="${MODEL_NAME}"
HP_SEARCH="${HP_SEARCH}"
ATL="${ATL:-0}" # default 0 for clf and plm
BATCH_SIZE="${BATCH_SIZE:-16}"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-instruct}"
TRAINING_SIZE="${TRAINING_SIZE:-5000}"

echo "Running with:"
echo "  LANG       = $LANG"
echo "  CONTEXT    = $CONTEXT"
echo "  MODEL_TYPE = $MODEL_TYPE"
echo "  ATL        = $ATL"
echo "  MODEL_NAME = $MODEL_NAME"
echo "  HP SEARCH = $HP_SEARCH"
echo "  BATCH_SIZE = $BATCH_SIZE"
echo "  PROMPT_TEMPLATE = $PROMPT_TEMPLATE"
echo "  TRAINING_SIZE = $TRAINING_SIZE"
echo

# VARS
EXPERIMENT="binary"
MAIN=1
SMOKE_TEST=0



# --------------------------------------------------------------------------------------------------
# HP SELECTION
# --------------------------------------------------------------------------------------------------

# HPs (single)
EPOCHS=3
LR_LIST=(2e-4)
BATCH_SIZE_LIST=("$BATCH_SIZE")
GRAD_NORM_LIST=(1)
WEIGHT_DECAY=0.01


if [ "$HP_SEARCH" -eq 1 ]; then
    if [ "$MODEL_TYPE" = "plm" ]; then
        # PLM HP search (12 combos)
        EPOCHS=3
        LR_LIST=(5e-5 1e-5 5e-6)
        BATCH_SIZE_LIST=(16 32)
        GRAD_NORM_LIST=(1)
        WEIGHT_DECAY=0.01
    else
        # SLM HP search (9 combos)
        EPOCHS=4
        LR_LIST=(5e-4 2e-4 5e-5)
        BATCH_SIZE_LIST=(16)
        GRAD_NORM_LIST=(0.5 1)
        WEIGHT_DECAY=0.01
    fi
fi


# --------------------------------------------------------------------------------------------------
# MODEL DIR
# --------------------------------------------------------------------------------------------------

if [[ "$MAIN" == "1" ]]; then
    MODEL_DIR="/scratch/prj/inf_nlg_ai_detection/wcd/data/exp1"
else 
    MODEL_DIR="/scratch/prj/inf_nlg_ai_detection/wcd/data/exp1_test"
fi

# Create a unique run dir
MODEL_LANG_DIR="${MODEL_DIR}/${LANG}"
mkdir -p "$MODEL_LANG_DIR"
RUN_DIR=$(mktemp -d "${MODEL_LANG_DIR}/run_XXXXXX")

echo "Run directory: $RUN_DIR"
echo


# --------------------------------------------------------------------------------------------------
# HP LOOP
# --------------------------------------------------------------------------------------------------

for BS in "${BATCH_SIZE_LIST[@]}"; do
  for LR in "${LR_LIST[@]}"; do
    for GN in "${GRAD_NORM_LIST[@]}"; do

      uv run run.py \
        --experiment "$EXPERIMENT" \
        --model_type "$MODEL_TYPE" \
        --model_name "$MODEL_NAME" \
        --prompt_template "$PROMPT_TEMPLATE" \
        --lang "$LANG" \
        --context "$CONTEXT" \
        --atl "$ATL" \
        --smoke_test "$SMOKE_TEST" \
        --training_size "$TRAINING_SIZE" \
        --run_dir "$RUN_DIR" \
        --epochs "$EPOCHS" \
        --learning_rate "$LR" \
        --batch_size "$BS" \
        --max_grad_norm "$GN" \
        --notes "testing 1 grad norm" \
        --weight_decay "$WEIGHT_DECAY"
        
    done
  done
done
