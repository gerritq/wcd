#!/bin/bash
#SBATCH --job-name=mask-eval
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --partition=nmes_gpu,gpu

set -euo pipefail

# Display GPU info
nvidia-smi

export PYTORCH_ALLOC_CONF=expandable_segments:True

# Configuration variables
LANGS=("en" "pt" "de" "ru" "it" "vi" "tr" "nl" "uk" "ro" "id" "bg" "uz" "no" "az" "mk" "hy" "sq")
MODEL_TYPE="slm"
MODEL_NAME="llama3_8b"
PROMPT_TEMPLATE="minimal"
CONTEXT=1
QUANTIZATION=1
MAX_LENGTH=512
SEED=42

echo "="
echo "Starting masked evaluation for ${#LANGS[@]} languages"
echo "Model: $MODEL_TYPE - $MODEL_NAME"
echo "="
echo

for LANG in "${LANGS[@]}"; do

    echo "="
    echo "Running masking_numbers.py evaluation with:"
    echo "  MODEL_TYPE      = $MODEL_TYPE"
    echo "  MODEL_NAME      = $MODEL_NAME"
    echo "  LANG            = $LANG"
    echo "  PROMPT_TEMPLATE = $PROMPT_TEMPLATE"
    echo "  CONTEXT         = $CONTEXT"
    echo "  QUANTIZATION    = $QUANTIZATION"
    echo "  MAX_LENGTH      = $MAX_LENGTH"
    echo "  SEED            = $SEED"
    echo "="
    echo

    # Run the evaluation script
    uv run masking_numbers.py \
        --model_type "$MODEL_TYPE" \
        --model_name "$MODEL_NAME" \
        --lang "$LANG" \
        --prompt_template "$PROMPT_TEMPLATE" \
        --context "$CONTEXT" \
        --quantization "$QUANTIZATION" \
        --max_length "$MAX_LENGTH" \
        --seed "$SEED"

    echo "="
    echo "Completed evaluation for $LANG"
    echo "="
    echo

done

echo "="
echo "All evaluations completed successfully!"
echo "="
