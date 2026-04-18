#!/bin/bash
set -euo pipefail

LANGS=("en" "it" "uk" "ro" "no" "az")
MODEL_TYPES=("clf") # slm clf plm
MODEL_NAMES=("llama3_8b")

ATL=(0)
CONTEXT=1
HP_SEARCH=1
TRAINING_SIZE=5000
SEED=42
EXPERIMENT="binary"
LANG_SETTING="random_csv"

TIME="10:00:00"

for lang in "${LANGS[@]}"; do
    for mtype in "${MODEL_TYPES[@]}"; do
        for mname in "${MODEL_NAMES[@]}"; do
            for atl in "${ATL[@]}"; do

                # skip classifier + atl=1 since it has no effect
                if [[ "$mtype" == "clf" && "$atl" -eq 1 ]]; then
                    echo "Skipping classifier with ATL=1"
                    continue
                fi

                job_name="e1-${mtype}-${lang}-random-c${CONTEXT}-atl${atl}-${mname}"

                echo "Submitting: $job_name (time=$TIME)"

                sbatch \
                    --job-name="$job_name" \
                    --time="$TIME" \
                    --export=ALL,LANG="$lang",CONTEXT="$CONTEXT",MODEL_TYPE="$mtype",ATL="$atl",MODEL_NAME="$mname",HP_SEARCH="$HP_SEARCH",TRAINING_SIZE="$TRAINING_SIZE",SEED="$SEED",EXPERIMENT="$EXPERIMENT",LANG_SETTING="$LANG_SETTING" \
                    exp1_job.sh
            done
        done
    done
done
