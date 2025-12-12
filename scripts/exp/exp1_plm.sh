#!/bin/bash
set -euo pipefail


# LANGS=("en" "nl" "it" "pt" "ru" "tr" "vi" "id" "ro" "uk" "bg" "no" "sq" "mk" "hy" "az")
LANGS=("de" "uz" "sq" "mk" "hy" "az")
CONTEXTS=(1)
MODEL_TYPES=("plm")
MODEL_NAMES=("mBert" "xlm-r-b" "xlm-r-l") # "mBert" "xlm-r-b" "xlm-r-l", "mDeberta-b", "mDeberta-l"
HP_SEARCH=1

TIME="00:30:00"

for lang in "${LANGS[@]}"; do
  for context in "${CONTEXTS[@]}"; do
    for mtype in "${MODEL_TYPES[@]}"; do
      for mname in "${MODEL_NAMES[@]}"; do

        job_name="e1-${mtype}-${lang}-c${context}-${mname}-HP${HP_SEARCH}"

        echo "Submitting: $job_name (time=$TIME)"

        sbatch \
          --job-name="$job_name" \
          --time="$TIME" \
          --export=ALL,LANG="$lang",CONTEXT="$context",MODEL_TYPE="$mtype",MODEL_NAME="$mname",HP_SEARCH="$HP_SEARCH" \
          exp1_job.sh

      done
    done
  done
done