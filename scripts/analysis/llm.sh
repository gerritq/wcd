#!/bin/bash
#SBATCH --job-name=llm
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=cpu,nmes_cpu
#SBATCH --mem=1GB

LANGUAGES=("en" "nl" "no" "it" "pt" "ro" "ru" "uk" "bg" "id")
# LANGUAGES=("en")
MODELS=("gpt-4o-mini")
SHOTS=0
SYSTEM=1

for LANGUAGE in "${LANGUAGES[@]}"; do
  for MODEL in "${MODELS[@]}"; do
      uv run llm.py \
            --lang "$LANGUAGE" \
            --model "$MODEL" \
            --shots "$SHOTS" \
            --system "$SYSTEM"

  done
done