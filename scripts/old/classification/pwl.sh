#!/bin/bash
#SBATCH --job-name=pwl-l8-gc
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=02:30:00
# SBATCH --partition=nmes_gpu,gpu
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --exclude=erc-hpc-comp039,erc-hpc-comp052

nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# LANGUAGES=("en" "nl" "no" "it" "pt" "ro" "ru" "uk" "bg" "id" "vi" "tr")
# LANGUAGES=("en" "nl" "no" "it" "pt" "ro")
# LANGUAGES=("ru" "uk" "bg" "id" "vi" "tr")
# LANGUAGES=("en" "nl" "no" "it" "pt" "ro")
LANGUAGES=("nl")

# llama3_8b
MODELS=("llama3_8b") # "llama3_8b" "qwen3_06b"

EPOCHS=3 # 3
BATCH=8
LEARNING_RATE=1e-4 # qlora paper 2e-4
GRAD_ACC=4
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=0.3
WARMUP_RATIO=0.03
SMOKE_TEST=0

for LANGUAGE in "${LANGUAGES[@]}"; do
  for MODEL in "${MODELS[@]}"; do

    uv run pwl.py \
      --lang "$LANGUAGE" \
      --model "$MODEL" \
      --learning_rate "$LEARNING_RATE" \
      --batch_size $BATCH \
      --epochs $EPOCHS \
      --grad_acc "$GRAD_ACC" \
      --warmup_ratio "$WARMUP_RATIO" \
      --weight_decay "$WEIGHT_DECAY" \
      --max_grad_norm "$MAX_GRAD_NORM" \
      --smoke_test $SMOKE_TEST
  done
done