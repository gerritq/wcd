#!/bin/bash
set -euo pipefail

LANGS=("id" "vi" "tr")

MODEL_TYPES=("slm") # slm clf
MODEL_NAMES=("llama3_8b") # "qwen3_8b"
TRAINING_SIZES=(200 400 600 800)

ATL=(1)
CONTEXT=1
HP_SEARCH=0

TIME="01:00:00"

for lang in "${LANGS[@]}"; do
    for mtype in "${MODEL_TYPES[@]}"; do
        for mname in "${MODEL_NAMES[@]}"; do
          for train_size in "${TRAINING_SIZES[@]}"; do
            for atl in "${ATL[@]}"; do

                # skip classifier + atl=1 since it has no effect
                if [[ "$mtype" == "clf" && "$atl" -eq 1 ]]; then
                    echo "Skipping classifier with ATL=1"
                    continue
                fi


                job_name="e1-${mtype}-${lang}-c${CONTEXT}-atl${atl}-${mname}-tran_size${train_size}"

                echo "Submitting: $job_name (time=$TIME)"

                sbatch \
                    --job-name="$job_name" \
                    --time="$TIME" \
                    --export=ALL,LANG="$lang",CONTEXT="$CONTEXT",MODEL_TYPE="$mtype",ATL="$atl",MODEL_NAME="$mname",HP_SEARCH="$HP_SEARCH",TRAINING_SIZE="$train_size" \
                    exp1_job.sh
                done
            done
       done
    done
done