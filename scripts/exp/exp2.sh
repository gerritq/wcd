#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --exclude=erc-hpc-comp050,erc-hpc-comp054,erc-hpc-comp039

# comp050 slow
# comp039 has error
nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_DIR="/scratch/prj/inf_nlg_ai_detection/wcd/data/exp2"

# get_run_number() {
#     dir="$1"
#     nums=$(ls -d "$dir"/run_* 2>/dev/null | sed 's/.*run_//' | grep -E '^[0-9]+$' || echo "")
#     if [ -z "$nums" ]; then
#         echo 1
#     else
#         echo $(( $(echo "$nums" | sort -n | tail -1) + 1 ))
#     fi
# }

# VARs
MODEL_TYPE="slm" # classifier
LANG="en"
ATL=1
ANNOTATION_TYPE="en"
MODEL_NAME="llama3_8b" # qwen3_06b llama3_8b qwen3_8b llama3_8b_base qwen3_8b_base
EXPLANATION="none"
TRAINING_SIZE_LIST=(500)
# TRAINING_SIZE_LIST=(500 1000 1500 2000) # -1 := full training
CONTEXT=1
QUANTIZATION=1
SMOKE_TEST=0
NOTES="vanilla hp search with en prompt"

# HPs
EPOCHS=2
LR_LIST=(2e-4)
BATCH_LIST=(24)
GRAD_NORM_LIST=(0.4)
WEIGHT_DECAY=0.01

# HPs (27 runs)
# EPOCHS=3
# LR_LIST=(5e-4 2e-4 5e-5)
# BATCH_LIST=(24)
# GRAD_NORM_LIST=(0.4 0.6 0.8)

# MODEL_LANG_DIR="${MODEL_DIR}/${LANG}"
# mkdir -p "$MODEL_LANG_DIR"
# NEXT_RUN=$(get_run_number "$MODEL_LANG_DIR")
# RUN_DIR="${MODEL_LANG_DIR}/run_${NEXT_RUN}"
# mkdir -p "$RUN_DIR"

for TRAINING_SIZE in "${TRAINING_SIZE_LIST[@]}"; do
  for LR in "${LR_LIST[@]}"; do
    for BS in "${BATCH_LIST[@]}"; do
      for GN in "${GRAD_NORM_LIST[@]}"; do

        uv run run.py \
          --model_type "$MODEL_TYPE" \
          --model_name "$MODEL_NAME" \
          --lang "$LANG" \
          --quantization "$QUANTIZATION" \
          --explanation "$EXPLANATION" \
          --context "$CONTEXT" \
          --smoke_test "$SMOKE_TEST" \
          --training_size "$TRAINING_SIZE" \
          --notes "$NOTES" \
          --epochs "$EPOCHS" \
          --learning_rate "$LR" \
          --batch_size "$BS" \
          --max_grad_norm "$GN" \
          --run_dir "$RUN_DIR" \
          --annotation_type "$ANNOTATION_TYPE" \
          --weight_decay "$WEIGHT_DECAY" \
          --atl "$ATL"

      done
    done
  done
done