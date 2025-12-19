#!/bin/bash
set -euo pipefail

# source ~/.bashrc
# cd /scratch/prj/inf_nlg_ai_detection/wcd/scripts/exp

# LANGS=("en" "pt" "de" "ru" "it" "vi" "tr" "nl" "uk" "ro" "id" "bg" "uz" "no" "az" "mk" "hy" "sq")
LANGS=("uk" "ro" "id" "bg" "uz" "no" "az" "mk" "hy" "sq")
MODEL_TYPES=("slm")
ATLS=(0 1)
MODEL_NAMES=("aya_8b") # "qwen3_8b" "llama3_8b" "aya_8b" "llama3_3b" "qwen3_4b"
PROMPT_TEMPLATES=("instruct") # ""minimal"  "instruct" "verbose"
BATCH_SIZE=16 # fixed for comparability

GPUS=2
CONTEXT=1
HP_SEARCH=1 # do not run HP search

for lang in "${LANGS[@]}"; do
  for mtype in "${MODEL_TYPES[@]}"; do
    for atl in "${ATLS[@]}"; do
      for mname in "${MODEL_NAMES[@]}"; do
        for ptemp in "${PROMPT_TEMPLATES[@]}"; do

          # skip classifier + atl=1 since it has no effect
          if [[ "$mtype" == "clf" && "$atl" -eq 1 ]]; then
            echo "Skipping classifier with ATL=1"
            continue
          fi

          # default time based on prompt template
          if [[ "$ptemp" == "verbose" ]]; then
              TIME="02:30:00"
          else
              TIME="02:30:00"
          fi

          # override if doing HP search
          if [[ "$HP_SEARCH" -eq 1 ]]; then
              if [[ "$ptemp" == "verbose" ]]; then
                  TIME="15:30:00"
              else
                  TIME="12:30:00"
              fi
          fi
          job_name="e1-${mtype}-${lang}-atl${atl}-${mname}-${ptemp}"

          echo "Submitting: $job_name (time=$TIME)"

          sbatch \
            --job-name="$job_name" \
            --time="$TIME" \
            --gres=gpu:="$GPUS" \
            --export=ALL,LANG="$lang",CONTEXT="$CONTEXT",MODEL_TYPE="$mtype",ATL="$atl",MODEL_NAME="$mname",HP_SEARCH="$HP_SEARCH",BATCH_SIZE="$BATCH_SIZE",PROMPT_TEMPLATE="$ptemp" \
            exp1_job.sh
            
        done
      done
    done
  done
done