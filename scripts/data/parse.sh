#!/bin/bash
#SBATCH --job-name=par-1
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=cpu,nmes_cpu
#SBATCH --mem=20GB

# "en" "nl" "no" "it"
# "pt" "ro" "ru" "uk"
# "bg" "id" "vi" "tr"
LANGUAGES=("id" "tr")

uv run parse.py --languages "${LANGUAGES[@]}"