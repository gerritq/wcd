#!/bin/bash
#SBATCH --job-name=sy-nl
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=cpu,nmes_cpu
#SBATCH --mem=20GB

LANG="nl"
MODEL="openai/gpt-4o-mini" # openai/gpt-5-mini openai/gpt-4o-mini
PROMPT="cot_wiki_short"
CONTEXT=1
TRAINING_SIZE=2000
SMOKE_TEST=0

uv run distil.py --lang "$LANG" \
                    --model "$MODEL" \
                    --prompt "$PROMPT" \
                    --context "$CONTEXT" \
                    --training_size "$TRAINING_SIZE" \
                    --smoke_test "$SMOKE_TEST"