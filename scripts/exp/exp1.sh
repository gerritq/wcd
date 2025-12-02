#!/bin/bash
#SBATCH --job-name=e1-atl-it-c1
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --exclude=erc-hpc-comp054,erc-hpc-comp040,erc-hpc-comp050

# comp050 slow
# comp039 has error

nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --------------------------------------------------------------------------------------------------
# SET VARS
# --------------------------------------------------------------------------------------------------

# VARs
MAIN=0
MODEL_TYPE="cls" # classifier slm plm cls
LANG="en"
ATL=1
CONTEXT=1

MODEL_NAME="llama3_8b" # qwen3_06b llama3_8b qwen3_8b mBert
TRAINING_SIZE=5000
SMOKE_TEST=0

EXPERIMENT="binary"

# --------------------------------------------------------------------------------------------------
# HP SELECTION
# --------------------------------------------------------------------------------------------------

# HPs (single)
EPOCHS=3
LR_LIST=(2e-4)
BATCH_SIZE_LIST=(24)
GRAD_NORM_LIST=(1)
WEIGHT_DECAY=0.01

if [ "$MODEL_TYPE" == "plm" ]; then
    # HPs PLMS (12 combos)
    EPOCHS=3
    LR_LIST=(5e-5 1e-5 5e-6)
    BATCH_SIZE_LIST=(16 32)    
    GRAD_NORM_LIST=(1)
    WEIGHT_DECAY=0.01
else
    # HPs SLM (9 combos)
    EPOCHS=3
    LR_LIST=(5e-4 2e-4 5e-5)
    BATCH_SIZE_LIST=(24)
    GRAD_NORM_LIST=(0.4 0.6 0.8)
    WEIGHT_DECAY=0.01
fi
    
    

# --------------------------------------------------------------------------------------------------
# MODEL DIR
# --------------------------------------------------------------------------------------------------

if [[ "$MAIN" == "1" ]]; then
    MODEL_DIR="/scratch/prj/inf_nlg_ai_detection/wcd/data/exp1"
else 
    MODEL_DIR="/scratch/prj/inf_nlg_ai_detection/wcd/data/exp1_test"
fi

# Create a uniqure run dir
MODEL_LANG_DIR="${MODEL_DIR}/${LANG}"
mkdir -p "$MODEL_LANG_DIR"
RUN_DIR=$(mktemp -d "${MODEL_LANG_DIR}/run_XXXXXX")


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
        --weight_decay "$WEIGHT_DECAY"
        
    done
  done
done