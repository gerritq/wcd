#!/bin/bash
#SBATCH --job-name=par-zh-th
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=cpu,nmes_cpu
#SBATCH --mem=20GB

# "en" "de" "id"
# "nl" "no" "it"
# "pt" "ro" "ru"
# "uk" "bg" "vi"
# "tr" "uz" "sq"
# "az" "mk" "hy"
LANGUAGES=("en" "de" "id")

uv run parse.py --languages "${LANGUAGES[@]}"