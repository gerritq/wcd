#!/bin/bash
#SBATCH --job-name=exp1
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=10:00:00
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

LANG="$1"
CONTEXT="$2"
MODEL_TYPE="$3"
ATL="$4"

echo "Running with:"
echo "  LANG       = $LANG"
echo "  CONTEXT    = $CONTEXT"
echo "  MODEL_TYPE = $MODEL_TYPE"
echo "  ATL        = $ATL"
echo

MAIN=1
MODEL_NAME="llama3_8b" # qwen3_06b llama3_8b qwen3_8b
TRAINING_SIZE=5000
SMOKE_TEST=0

EXP_N=1
QUANTIZATION=1
PROMPT_EXTENSION="" # do not add "_"
NOTES=""

# HPs (search - 9 combos)
EPOCHS=3
LR_LIST=(5e-4 2e-4 5e-5)
BATCH_SIZE=24
GRAD_NORM_LIST=(0.4 0.6 0.8)
WEIGHT_DECAY=0.01


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


for LR in "${LR_LIST[@]}"; do
    for GN in "${GRAD_NORM_LIST[@]}"; do

      echo ">>> Starting run with LR=$LR, max_grad_norm=$GN"
      uv run run.py \
        --model_type "$MODEL_TYPE" \
        --model_name "$MODEL_NAME" \
        --lang "$LANG" \
        --quantization "$QUANTIZATION" \
        --context "$CONTEXT" \
        --smoke_test "$SMOKE_TEST" \
        --training_size "$TRAINING_SIZE" \
        --notes "$NOTES" \
        --epochs "$EPOCHS" \
        --learning_rate "$LR" \
        --batch_size "$BATCH_SIZE" \
        --max_grad_norm "$GN" \
        --run_dir "$RUN_DIR" \
        --weight_decay "$WEIGHT_DECAY" \
        --atl "$ATL" \
        --prompt_extension "$PROMPT_EXTENSION" \
        --experiment_number "$EXP_N"

      echo "<<< Finished run with LR=$LR, max_grad_norm=$GN"
      echo
    done
done