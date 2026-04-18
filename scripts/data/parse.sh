#!/bin/bash
#SBATCH --job-name=par-en-it
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=cpu,nmes_cpu
#SBATCH --mem=20GB

# "en" "de" 
# "id" "nl" 
# "no" "it"
# "pt" "ro"
# "ru" "uk"
# "bg" "vi"
# "tr" "uz" 
# "sq" "az"
# "mk" "hy"
LANGUAGES=("uk" "ro" "no" "az")

uv run parse.py --languages "${LANGUAGES[@]}"