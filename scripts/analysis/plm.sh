#!/bin/bash
#SBATCH --job-name=plm-nl
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# "en" "nl" "no" "it" "pt" "ro" "ru" "uk" "bg"
LANGUAGES=("en" "nl" "no")
MODELS=("mBert") # "xlm-r-b" "mBert" "mBert"
CONTEXT=1

# HP
EPOCH=2
LEARNING_RATE=1e-05

for LANGUAGE in "${LANGUAGES[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    echo "Running with LANGUAGE=$LANGUAGE, MODEL=$MODEL"
    start=$(date +%s)

    uv run plm.py \
      --lang "$LANGUAGE" \
      --model "$MODEL" \
      --context $CONTEXT \
      --epochs $EPOCH \
      --learning_rate "$LEARNING_RATE"

    end=$(date +%s)
    runtime=$((end - start))
    echo "Time taken for LANGUAGE=$LANGUAGE, MODEL=$MODEL: $((runtime / 60))m"
  done
done
