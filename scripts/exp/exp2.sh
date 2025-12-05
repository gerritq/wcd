#!/bin/bash
#SBATCH --job-name=exp4-atl-enru-bg
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --exclude=erc-hpc-comp054,erc-hpc-comp040

# comp050 slow
# comp039 has error
nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# VARs
MODEL_TYPE="slm" # classifier slm
TEST_LANG="bg"
TRAINING_LANGS=("en")
LANG="$TEST_LANG"
ATL=1

MODEL_NAME="llama3_8b" # qwen3_06b llama3_8b qwen3_8b mBert
TRAINING_SIZE_LIST=(1000)
SMOKE_TEST=0

CONTEXT=1
EXP="cl"
QUANTIZATION=1
PROMPT_TEMPLATE="instruct"


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
    

for TRAINING_SIZE in "${TRAINING_SIZE_LIST[@]}"; do
  for LR in "${LR_LIST[@]}"; do
    for BS in "${BATCH_SIZE_LIST[@]}"; do
      for GN in "${GRAD_NORM_LIST[@]}"; do

        uv run run.py \
          --model_type "$MODEL_TYPE" \
          --model_name "$MODEL_NAME" \
          --prompt_template "$PROMPT_TEMPLATE" \
          --lang "$LANG" \
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
done