#!/bin/bash
#SBATCH --job-name=r-en
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=20GB
#SBATCH --partition=cpu,nmes_cpu

# "en" "de" 
# "id" "nl" 
# "no" "it"
# "pt" "ro"
# "ru" "uk"
# "bg" "vi"
# "tr" "uz" 
# "sq" "az"
# "mk" "hy"
LANGUAGES=("pt" "bg" "tr" "mk" "ru")

# "en" "it" "uk" "ro" "no" "az"

uv run raw.py --languages "${LANGUAGES[@]}"