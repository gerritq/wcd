#!/bin/bash
#SBATCH --job-name=table1
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=nmes_cpu,cpu
#SBATCH --mem=10GB

uv run size.py \
    --lang nl \
    --context 1 \
    --explanation 1 \
    --model_type atl \
    --model_name llama3_8b