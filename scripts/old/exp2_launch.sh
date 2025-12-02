#!/bin/bash
set -euo pipefail

# Grids
MODEL_TYPES=("slm")
LANGS=("it")
ATLS=(0)

for mtype in "${MODEL_TYPES[@]}"; do
  for lang in "${LANGS[@]}"; do
    for atl in "${ATLS[@]}"; do

      # Skip invalid combo: classifier + ATL=1
      if [[ "$mtype" == "classifier" && "$atl" -eq 1 ]]; then
        echo "Skipping: MODEL_TYPE=classifier with ATL=1 (not allowed)"
        continue
      fi

      job_name="e2-${mtype}-${lang}-atl${atl}"
      echo "Submitting job: ${job_name}"

      sbatch --job-name="$job_name" exp2_job.sh \
        "$mtype" \
        "$lang" \
        "$atl"

    done
  done
done