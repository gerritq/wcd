#!/bin/bash
#SBATCH --job-name=topic
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=cpu,nmes_cpu
#SBATCH --mem=20GB

LANGUAGES=("en" "nl" "no" "it" "pt" "ro" "ru" "uk" "bg" "id" "id" "vi" "tr")

for LANGUAGE in "${LANGUAGES[@]}"; do
    uv run topic_plot.py \
    --lang "$LANGUAGE"
done
