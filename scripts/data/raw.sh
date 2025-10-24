#!/bin/bash
#SBATCH --job-name=r-3
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=05:00:00
#SBATCH --mem=20GB
#SBATCH --partition=cpu,nmes_cpu

# "en" "nl" "no" "it"
# "pt" "ro" "ru" "uk"
# "bg" "id" "vi" "tr"
LANGUAGES=("bg" "id" "vi" "tr")

uv run raw.py --languages "${LANGUAGES[@]}"