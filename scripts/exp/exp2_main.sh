#!/bin/bash
set -euo pipefail

source ~/.bashrc
cd /scratch/prj/inf_nlg_ai_detection/wcd/scripts/exp


MODEL_TYPE="slm"
MODEL_NAMES=("mBert")
ATL=(0)
SEEDS="42"

# all med and lower langs
TARGET_LANGS="uk ro id bg uz no az mk hy sq"
SOURCE_LANGS="en"
LANG="en"
LANG_SETTINGS="main"
CL_SETTINGS="few"

EXPERIMENT="cl_eval"

TIME="01:00:00"


for mname in "${MODEL_NAMES[@]}"; do

    # skip classifier + atl=1 since it has no effect
    if [[ ("$mtype" == "clf" || "$mtype" == "plm") && "$atl" -eq 1 ]]; then
        echo "Skipping classifier with ATL=1"
        continue
    fi

    job_name="e2-${mtype}-${mname}"

    echo "Submitting: $job_name (time=$TIME)"

    sbatch \
    --job-name="$job_name" \
    --time="$TIME" \
    --export=ALL,LANG="$lang",MODEL_TYPE="$mtype",ATL="$atl",MODEL_NAME="$mname",HP_SEARCH="$HP_SEARCH",CONTEXT="$CONTEXT",SEED="$seed",EXPERIMENT="$EXPERIMENT" \
    exp1_job.sh
done