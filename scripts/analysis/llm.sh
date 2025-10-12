#!/bin/bash
#SBATCH --job-name=llm
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=cpu,nmes_cpu
#SBATCH --mem=1GB
#SBATCH --gres=gpu:1

LANGUAGE="en"
MODEL="openai/gpt-4o-mini"

uv run llm.py \
      --lang "$LANGUAGE" \
      --model "$MODEL" 