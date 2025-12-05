#!/bin/bash
set -euo pipefail

LANGS=("en_ct24" "ar_ct24" "nl_ct24" "en_nlp4if" "ar_nlp4if" "bg_nlp4if")
# LANGS=("en_ct24" "ar_ct24" "nl_ct24" "ar_nlp4if" "bg_nlp4if")
MODEL_TYPES=("slm" "clf") # slm clf
ATLS=(0 1)
MODEL_NAMES=("llama3_8b") # "qwen3_8b"
CONTEXT=0 # no context for external data
HP_SEARCH=0 # do not run hp search

for lang in "${LANGS[@]}"; do
    for mtype in "${MODEL_TYPES[@]}"; do
        for atl in "${ATLS[@]}"; do
        for mname in "${MODEL_NAMES[@]}"; do

            # skip classifier + atl=1 since it has no effect
            if [[ "$mtype" == "clf" && "$atl" -eq 1 ]]; then
                echo "Skipping classifier with ATL=1"
                continue
            fi

            # select time
            if [[ "$lang" == "ar_ct24" || "$lang" == "en_ct24"  ]]; then
                TIME="03:00:00"
            else
                TIME="01:00:00"
            fi

            job_name="e1-${mtype}-${lang}-atl${atl}-${mname}"

            echo "Submitting: $job_name (time=$TIME)"

            sbatch --job-name="$job_name" \
                    --time="$TIME" \
                    exp1_job.sh \
                    "$lang" \
                    "$CONTEXT" \
                    "$mtype" \
                    "$atl" \
                    "$mname" \
                    "$HP_SEARCH"

        done
        done
    done
done