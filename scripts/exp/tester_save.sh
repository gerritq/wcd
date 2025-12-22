#!/bin/bash
set -euo pipefail

# ---- export variables (sbatch-style replacement) ----
export SOURCE_LANGS="en"
export LANG="en"
export MODEL_TYPE="clf"
export ATL="1"
export MODEL_NAME="llama3_8b"
export HP_SEARCH="0"
export CONTEXT="1"
export SEED="42"
export EXPERIMENT="save"

echo "Running exp1_job.sh with:"
env | grep -E 'LANG=|MODEL_TYPE=|ATL=|MODEL_NAME=|HP_SEARCH=|CONTEXT=|SEED=|EXPERIMENT='
echo

# ---- run the job script ----
sbatch exp1_job.sh