#!/bin/bash
set -euo pipefail

LANGS=("en" "nl" "it" "pt" "ru" "tr" "vi" "id" "ro" "uk" "bg" "no" "sq" "mk" "hy" "az" "de" "uz")
MODEL_TYPES=("clf")
MODEL_NAMES=("qwen3_8b" "llama3_8b" "aya_8b") #  llama_8b

CONTEXT=1
HP_SEARCH=1

TIME="08:00:00"

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