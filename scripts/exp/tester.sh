#!/bin/bash
set -euo pipefail

# ---- export variables (sbatch-style replacement) ----
export LANG="en"
export MODEL_TYPE="slm"
export ATL="0"
export MODEL_NAME="llama3_8b"
export HP_SEARCH="0"
export CONTEXT="1"
export SEED="2025"
export EXPERIMENT="seed"

echo "Running exp1_job.sh with:"
env | grep -E 'LANG=|MODEL_TYPE=|ATL=|MODEL_NAME=|HP_SEARCH=|CONTEXT=|SEED=|EXPERIMENT='
echo

# ---- run the job script ----
bash exp1_job.sh