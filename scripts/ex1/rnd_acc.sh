#!/bin/bash
#SBATCH --job-name=rnd-acc-l8-en
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --exclude=erc-hpc-comp039,erc-hpc-comp054

nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LANGUAGES=("en")
MODELS=("llama3_8b")
TRIALS=1
SMOKE_TEST=0
PWL=1

for LANGUAGE in "${LANGUAGES[@]}"; do
  for MODEL in "${MODELS[@]}"; do

    uv run rnd_acc.py \
      --lang "$LANGUAGE" \
      --model "$MODEL" \
      --trials $TRIALS \
      --pwl $PWL \
      --smoke_test $SMOKE_TEST
  done
done