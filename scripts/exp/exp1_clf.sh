#!/bin/bash
set -euo pipefail

LANGS=("ru" "uk" "id" "vi" "tr")
MODEL_TYPES=("clf")
MODEL_NAMES=("llama3_8b")

CONTEXT=1
HP_SEARCH=0

TIME="01:30:00"

for lang in "${LANGS[@]}"; do
  for mtype in "${MODEL_TYPES[@]}"; do
    for mname in "${MODEL_NAMES[@]}"; do

      job_name="e1-${mtype}-${lang}-${mname}"

      echo "Submitting: $job_name (time=$TIME)"

      sbatch \
        --job-name="$job_name" \
        --time="$TIME" \
        --export=ALL,LANG="$lang",CONTEXT="$CONTEXT",MODEL_TYPE="$mtype",MODEL_NAME="$mname",HP_SEARCH="$HP_SEARCH" \
        exp1_job.sh

    done
  done
done