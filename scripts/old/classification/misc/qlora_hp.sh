#!/bin/bash
#SBATCH --job-name=q06B
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

set -e
nvidia-smi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ==================== Create model dir ====================
# Find and create the model_\d dir to be passed to the script for this run
MODEL_DIR="/scratch/prj/inf_nlg_ai_detection/wcd/data/models/slm/hp" 
# find existing models: find all models, get their names, sort them in desc order
existing_models=($(ls -d "$MODEL_DIR"/model_* 2>/dev/null | grep -Eo 'model_[0-9]+' | sort -V))
# if array is empty, create model_1; else pick the largest number and increment
if [ ${#existing_models[@]} -eq 0 ]; then
    next_num=1
else
    last_model="${existing_models[-1]}"
    last_num=$(echo "$last_model" | grep -Eo '[0-9]+$')
    next_num=$((last_num + 1))
fi
# create the model dir, to be passed to the py file
model_dir="${MODEL_DIR}/model_${next_num}"
model_dir_best_model="${model_dir}/best_model"
model_dir_temp="${model_dir}/temp"
mkdir -p "$model_dir"
mkdir -p "$model_dir_best_model"
mkdir -p "$model_dir_temp"
touch "${model_dir}/meta_overview.jsonl"
echo "Created $model_dir and subfiles"

# ==================== Run HP tuning ====================
# Langs and models
LANGUAGES=("en")
MODELS=("llama3_1b")
PLW=0

# HP search space
BATCH=8
GRADIENT_ACCUMULATION_STEPS=(2)
# EPOCHS=(1 3)
# LEARNING_RATE=(2e-4 2e-5 2e-6)
# LORA_R=(16 32 64)  # as recommended by unsloth (r = alpha), so we do not tune this

EPOCHS=(1)
LEARNING_RATE=(2e-4 2e-6)
LORA_R=(32)  # as recommended by unsloth (r = alpha), so we do not tune this

start_time=$(date +%s)

for LANGUAGE in "${LANGUAGES[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for GA in "${GRADIENT_ACCUMULATION_STEPS[@]}"; do
      for EPOCH in "${EPOCHS[@]}"; do
        for LR in "${LEARNING_RATE[@]}"; do
          for R in "${LORA_R[@]}"; do

            uv run qlora_hp.py \
              --model_dir "$model_dir" \
              --lang "$LANGUAGE" \
              --model "$MODEL" \
              --learning_rate "$LR" \
              --batch_size "$BATCH" \
              --epochs "$EPOCH" \
              --grad_accum "$GA" \
              --lora_r "$R" \
              --plw "$PLW"

          done
        done
      done
    done
  done
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Runtime: $(($elapsed / 60)) min"