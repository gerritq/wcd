#!/bin/bash
#SBATCH --job-name=exp2-clf-no-en
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --exclude=erc-hpc-comp054,erc-hpc-comp040,erc-hpc-comp050

# comp050 slow
# comp039 has error
nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# -------------------------------------------------------------------
# EXP2 VARs
# -------------------------------------------------------------------
MODEL_TYPE="clf" # classifier slm clf
TEST_LANG="no"
TRAINING_LANGS=("en")
ATL=0

MODEL_NAME="llama3_8b" # qwen3_06b llama3_8b qwen3_8b mBert
TRAINING_SIZE=5000

EXP="cl"
SMOKE_TEST=0
CONTEXT=1
QUANTIZATION=1
PROMPT_TEMPLATE="instruct"

# -------------------------------------------------------------------
# HP selection
# -------------------------------------------------------------------

if [ "$MODEL_TYPE" == "plm" ]; then
    # HPs PLMS (12 combos)
    EPOCHS=3
    LR_LIST=(5e-5)
    BATCH_SIZE_LIST=(32)    
    GRAD_NORM_LIST=(1)
    WEIGHT_DECAY=0.01
else
    # HPs SLM (9 combos)
    EPOCHS=3
    LR_LIST=(2e-4)
    BATCH_SIZE_LIST=(24)
    GRAD_NORM_LIST=(0.4)
    WEIGHT_DECAY=0.01
fi
    

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