#!/bin/bash
#SBATCH --job-name=aya
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# "xlm-r": "FacebookAI/xlm-roberta-base",
# "mBert": "google-bert/bert-base-multilingual-uncased",
# "llama3_1b": "meta-llama/Llama-3.2-1B-Instruct",
# "llama3_3b": "meta-llama/Llama-3.2-3B-Instruct",
# "llama3_8b": "meta-llama/Llama-3.1-8B-Instruct",
# "llama3_70b": "meta-llama/Llama-3.3-70B-Instruct",
# "qwen_06b": "Qwen/Qwen3-0.6B",
# "qwen3_4b": "Qwen/Qwen3-4B-Instruct-2507",
# "qwen3_8b": "Qwen/Qwen3-8B",
# "qwen3_30b": "Qwen/Qwen3-30B-A3B-Instruct-2507",
# "qwen3_32b": "Qwen/Qwen3-32B"

# LANGUAGES=("en" "nl" "no" "it" "pt" "ro" "ru" "uk" "bg" "id" "multi")
LANGUAGES=("en") #  "it" "ru" "multi"
 
# LANGUAGES=("multi")
MODELS=(
  "aya"
)


BATCH=16
EPOCHS=5
PLW=0
SYSTEM=1
NOTES=""

for LANGUAGE in "${LANGUAGES[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "Running with MODEL=$MODEL, LANGUAGES=$LANGUAGE"
    uv run peft_s2s.py \
      --lang "$LANGUAGE" \
      --model "$MODEL" \
      --batch_size $BATCH \
      --epochs $EPOCHS \
      --plw $PLW \
      --system "$SYSTEM" \
      --notes "$NOTES"
  done
done