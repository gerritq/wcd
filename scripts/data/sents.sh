#!/bin/bash
#SBATCH --job-name=s-1
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=02:30:00
# SBATCH --partition=cpu,nmes_cpu
#SBATCH --partition=nmes_gpu,gpu,interruptible_gpu
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

# Takes about 2hrs for English. Other languages should take less time.

nvidia-smi

# "en" "nl" "no"
# "it" "pt" "ro"
# "ru" "uk" "bg"
# "id" "vi" "tr"
LANGUAGES=("en")

start=$(date +%s)

uv run sents.py --languages "${LANGUAGES[@]}"

end=$(date +%s)
runtime=$((end - start))
echo "Runtime: $(echo "scale=2; $runtime/60" | bc) minutes"