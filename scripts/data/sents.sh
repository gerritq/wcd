#!/bin/bash
#SBATCH --job-name=s-4
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=06:30:00
# SBATCH --partition=cpu,nmes_cpu
#SBATCH --partition=nmes_gpu,gpu,interruptible_gpu
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --exclude=erc-hpc-comp039,erc-hpc-comp040
# Takes about 2hrs for English. Other languages should take less time.

nvidia-smi

# "en" "nl" "no"
# "it" "pt" "ro"
# "ru" "uk" "bg"
# "id" "vi" "tr"
LANGUAGES=("nl" "it" "ro")

start=$(date +%s)

uv run sents.py --languages "${LANGUAGES[@]}"

end=$(date +%s)
runtime=$((end - start))
echo "Runtime: $(echo "scale=2; $runtime/60" | bc) minutes"