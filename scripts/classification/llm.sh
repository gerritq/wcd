#!/bin/bash
#SBATCH --job-name=llm
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=cpu,nmes_cpu
#SBATCH --mem=1GB

# LANGUAGES=("en" "nl" "no" "it" "pt" "ro" "ru" "uk" "bg" "id")
LANGUAGES=("uk" "bg" "id")
MODELS=("gemini-2.5-flash-lite" "gpt-4o-mini") # google/gemini-2.5-flash-lite gpt-4o-mini
SHOTS=(0 1)
VERBOSE=(0)

SYSTEM=1

for LANGUAGE in "${LANGUAGES[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for SHOT in "${SHOTS[@]}"; do
      for VERB in "${VERBOSE[@]}"; do
        uv run llm.py \
          --lang "$LANGUAGE" \
          --model "$MODEL" \
          --shots "$SHOT" \
          --system "$SYSTEM" \
          --verbose "$VERB"
      done
    done
  done
done