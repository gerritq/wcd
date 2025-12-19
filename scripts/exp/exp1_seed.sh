#!/bin/bash
set -euo pipefail

source ~/.bashrc
cd /scratch/prj/inf_nlg_ai_detection/wcd/scripts/exp1


LANGS=("no" "ro" "bg" "tr")
CONTEXTS=(1)
MODEL_TYPES=("clf") # slm clf
ATLS=(0 1)
MODEL_NAMES=("llama3_8b") # "qwen3_8b"
SEEDS(2025 2026)

TIME="02:00:00"

for lang in "${LANGS[@]}"; do
  for ctx in "${CONTEXTS[@]}"; do
    for mtype in "${MODEL_TYPES[@]}"; do
      for atl in "${ATLS[@]}"; do
        for mname in "${MODEL_NAMES[@]}"; do

          # skip classifier + atl=1 since it has no effect
          if [[ "$mtype" == "clf" && "$atl" -eq 1 ]]; then
            echo "Skipping classifier with ATL=1"
            continue
          fi

          # select time
          if [[ "$mtype" == "clf" ]]; then
            TIME="05:00:00"
          else
            TIME="10:00:00"
          fi

          job_name="e1-${mtype}-${lang}-c${ctx}-atl${atl}-${mname}"

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
done