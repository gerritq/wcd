#!/bin/bash
set -euo pipefail

LANGS=("it" "nl")
CONTEXTS=(0 1)
MODEL_TYPES=("classifier" "slm")
ATLS=(0 1)

for lang in "${LANGS[@]}"; do
  for ctx in "${CONTEXTS[@]}"; do
    for mtype in "${MODEL_TYPES[@]}"; do
      for atl in "${ATLS[@]}"; do

        # skip classifier and atl 1 as this has noe effect
        if [[ "$mtype" == "classifier" && "$atl" -eq 1 ]]; then
          echo "Skipping classifier with ATL=1"
          continue
        fi

        job_name="e1-${mtype}-${lang}-c${ctx}-atl${atl}"
        
        sbatch --job-name="$job_name" exp1_job.sh \
          "$lang" \
          "$ctx" \
          "$mtype" \
          "$atl"

      done
    done
  done
done