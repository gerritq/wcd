#!/bin/bash
set -euo pipefail

source ~/.bashrc
cd /scratch/prj/inf_nlg_ai_detection/wcd/scripts/exp



LANGS=("en" "nl" "no" "it" "pt" "ro" "ru" "uk" "bg" "id" "vi" "tr")
CONTEXTS=(1)
MODEL_TYPES=("plm")
MODEL_NAMES=("mBert" "xlm-r-b" "xlm-r-l" "mDeberta-b" "mDeberta-l") # "xlm-r-b" "xlm-r-l", "mDeberta-b", "mDeberta-l"
atl=0   # ATL has no effect for PLMs

TIME="02:00:00"

for lang in "${LANGS[@]}"; do
  for ctx in "${CONTEXTS[@]}"; do
    for mtype in "${MODEL_TYPES[@]}"; do
      for mname in "${MODEL_NAMES[@]}"; do

        job_name="e1-${mtype}-${lang}-c${ctx}-${mname}"

        echo "Submitting: $job_name (time=$TIME)"

        sbatch --job-name="$job_name" \
               --time="$TIME" \
               exp1_job.sh \
               "$lang" \
               "$ctx" \
               "$mtype" \
               "$atl" \
               "$mname"

      done
    done
  done
done