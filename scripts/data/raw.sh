#!/bin/bash
#SBATCH --job-name=r-zh-th
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=20GB
#SBATCH --partition=cpu,nmes_cpu

# 1 # "en" "nl" "no" "it"
# 2 # "pt" "ro" "ru" "uk"
# 3 # "sr" "bg" "id" "vi" "tr"
# "sq" "az" "mk" "hy"
LANGUAGES=("zh" "th")

uv run raw.py --languages "${LANGUAGES[@]}"