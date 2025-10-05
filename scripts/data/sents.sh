#!/bin/bash
#SBATCH --job-name=sents
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=cpu,nmes_cpu
#SBATCH --mem=20GB

langs=("en" "hu" "pl" "pt")

for lang in "${langs[@]}"; do
    echo "Running for $lang..."
    uv run sents.py $lang
done