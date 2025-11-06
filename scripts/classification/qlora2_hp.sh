#!/bin/bash
#SBATCH --job-name=ray-hp
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --exclude=erc-hpc-comp039,erc-hpc-comp054

nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook

LANGUAGES=("nl")
MODELS=("llama3_8b")
TRIALS=15
SMOKE_TEST=0

for LANGUAGE in "${LANGUAGES[@]}"; do
  for MODEL in "${MODELS[@]}"; do

    uv run --active qlora2_hp.py \
      --lang "$LANGUAGE" \
      --model "$MODEL" \
      --trials $TRIALS \
      --smoke_test $SMOKE_TEST
  done
done