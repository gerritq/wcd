#!/bin/bash
set -euo pipefail

source ~/.bashrc
cd /scratch/prj/inf_nlg_ai_detection/wcd/scripts/exp

# LANGS=("en" "pt" "de" "ru" "it" "vi" "tr" "nl" "uk" "ro" "id" "bg" "uz" "no" "az" "mk" "hy" "sq")
LANGS=("bg")
MODEL_TYPES=("clf") # slm clf plm
ATLS=(0)
MODEL_NAMES=("llama3_8b") # "qwen3_8b" "llama3_8b" "aya_8b"
# MODEL_NAMES=("mBert" "xlm-r-b" "xlm-r-l")
SEEDS=(2025)

BATCH_SIZE=8 # lower to avoid cuda errors
HP_SEARCH=0
CONTEXT=1
EXPERIMENT="seed"

TIME="01:30:00"

for lang in "${LANGS[@]}"; do
    for mtype in "${MODEL_TYPES[@]}"; do
      for atl in "${ATLS[@]}"; do
        for mname in "${MODEL_NAMES[@]}"; do
          for seed in "${SEEDS[@]}"; do

          # skip classifier + atl=1 since it has no effect
          if [[ "$mtype" == "clf" && "$atl" -eq 1 ]]; then
            echo "Skipping classifier with ATL=1"
            continue
          fi

          job_name="e1-${mtype}-${lang}-c${CONTEXT}-atl${atl}-${mname}-seed${seed}"

          echo "Submitting: $job_name (time=$TIME)"

          sbatch \
            --job-name="$job_name" \
            --time="$TIME" \
            --export=ALL,LANG="$lang",MODEL_TYPE="$mtype",ATL="$atl",MODEL_NAME="$mname",HP_SEARCH="$HP_SEARCH",CONTEXT="$CONTEXT",SEED="$seed",EXPERIMENT="$EXPERIMENT",BATCH_SIZE="$BATCH_SIZE" \
            exp1_job.sh

        done
      done
    done
  done
done