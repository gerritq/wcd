#!/bin/bash
set -euo pipefail

# source ~/.bashrc
# cd /scratch/prj/inf_nlg_ai_detection/wcd/scripts/exp

# If two langs combine with +

TRAINING_LANGS=("en")
MODEL_TYPES=("clf")
ATLS=(0)
MODEL_NAMES=("llama3_8b") # "qwen3_8b" "llama3_8b" "aya_8b" "llama3_3b" "qwen3_4b"
SEEDS=(2025 2026)

PROMPT_TEMPLATES=("instruct") # ""minimal"  "instruct" "verbose"
BATCH_SIZE=16 # fixed for comparability

GPUS=1
CONTEXT=1
HP_SEARCH=0 # do not run HP search
EXPERIMENT="save"

for lang in "${TRAINING_LANGS[@]}"; do
  for mtype in "${MODEL_TYPES[@]}"; do
    for atl in "${ATLS[@]}"; do
      for mname in "${MODEL_NAMES[@]}"; do
        for seed in "${SEEDS[@]}"; do

            # skip classifier + atl=1 since it has no effect
            if [[ ("$mtype" == "clf" || "$mtype" == "plm")  && "$atl" -eq 1 ]]; then
            echo "Skipping classifier/plm with ATL=1"
            continue
            fi

            # default time based on prompt template
            if [[ "$mtype" == "plm" ]]; then
                TIME="00:15:00"
            else
                TIME="02:00:00"
            fi

        job_name="e1-${mtype}-trainon[${lang}]-atl${atl}-${mname}-${seed}"

          echo "Submitting: $job_name (time=$TIME)"

          sbatch \
            --job-name="$job_name" \
            --time="$TIME" \
            --gres=gpu:"$GPUS" \
            --export=ALL,MODEL_TYPE="$mtype",ATL="$atl",MODEL_NAME="$mname",HP_SEARCH="$HP_SEARCH",CONTEXT="$CONTEXT",SEED="$seed",EXPERIMENT="$EXPERIMENT",SOURCE_LANGS="$lang",LANG="$lang" \
            exp1_job.sh
            
        done
      done
    done
  done
done