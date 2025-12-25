#!/bin/bash
set -euo pipefail

source ~/.bashrc
cd /scratch/prj/inf_nlg_ai_detection/wcd/scripts/exp


MODEL_TYPE="slm"
MODEL_NAMES=("llama3_8b")
ATL=1
SEEDS="42"

# all med and lower langs
# TARGET_LANGS="uk ro id bg uz no az mk hy sq"
# TARGET_LANGS="uk ro id bg uz"
TARGET_LANGS="uk ro id bg uz no az mk hy sq"
SOURCE_LANGS="en"

LANG_SETTINGS="main translation"
CL_SETTINGS="few"

LOWER_LR=1
NEW_LEARNING_RATE=1e-6 # 5e-6 1e-6

BATCH_SIZE=16
EXPERIMENT="cl_eval"

TIME="15:00:00"

for mname in "${MODEL_NAMES[@]}"; do

    # skip classifier + atl=1 since it has no effect
    if [[ ("$MODEL_TYPE" == "clf" || "$MODEL_TYPE" == "plm") && "$ATL" -eq 1 ]]; then
        echo "Skipping classifier with ATL=1"
        continue
    fi

    TARGET_TAG=$(echo "$TARGET_LANGS" | tr ' ' '+')
    job_name="e2-${MODEL_TYPE}-${mname}-${TARGET_TAG}-lr${NEW_LEARNING_RATE}"

    echo "Submitting: $job_name (time=$TIME)"

    sbatch \
    --job-name="$job_name" \
    --time="$TIME" \
    --export=ALL,LANG="$SOURCE_LANGS",MODEL_TYPE="$MODEL_TYPE",ATL="$ATL",MODEL_NAME="$mname",SEEDS="$SEEDS",EXPERIMENT="$EXPERIMENT",TARGET_LANGS="$TARGET_LANGS",SOURCE_LANGS="$SOURCE_LANGS",LANG_SETTINGS="$LANG_SETTINGS",CL_SETTINGS="$CL_SETTINGS",LOWER_LR="$LOWER_LR",NEW_LEARNING_RATE="$NEW_LEARNING_RATE" \
    exp2_job.sh
done