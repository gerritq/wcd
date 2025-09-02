#!/bin/bash
#SBATCH --job-name=pt
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu,nmes_gpu,interruptible_gpu
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

nvidia-smi

DATA="pl_sents" # ours_en cn_fa cn_fa_ss cn_fa_ss_nl
MODEL="bert-base-multilingual-uncased" # bert-base-uncased
N=5000
HP_SEARCH="" # "--hp_search"
NOTES="first hu run"

uv run cd_hp.py \
    --data "$DATA" \
    --model "$MODEL" \
    --n "$N" \
    --notes "$NOTES" \
    $HP_SEARCH