#!/bin/bash
set -euo pipefail

# source ~/.bashrc
# cd /scratch/prj/inf_nlg_ai_detection/wcd/scripts/exp


# LANGS=("en" "nl" "it" "pt" "ru" "tr" "vi" "id" "ro" "uk" "bg" "no" "sq" "mk" "hy" "az")
LANGS=("de" "uz")
MODEL_TYPES=("slm")
ATLS=(0 1)
MODEL_NAMES=("qwen3_8b" "llama_8b" "aya_8b") # qwen3_8b llama_8b aya_8b
PROMPT_TEMPLATES=("minimal") # ""minimal"  "instruct" "verbose"
BATCH_SIZE=16 # fixed for comparability

CONTEXT=1
HP_SEARCH=0 # do not run HP search

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
              TIME="01:30:00"
          fi

          # override if doing HP search
          if [[ "$HP_SEARCH" -eq 1 ]]; then
              if [[ "$ptemp" == "verbose" ]]; then
                  TIME="15:30:00"
              else
                  TIME="10:30:00"
              fi
          fi
          job_name="e1-${mtype}-${lang}-atl${atl}-${mname}-${ptemp}"

          echo "Submitting: $job_name (time=$TIME)"

          sbatch \
            --job-name="$job_name" \
            --time="$TIME" \
            --export=ALL,LANG="$lang",CONTEXT="$CONTEXT",MODEL_TYPE="$mtype",ATL="$atl",MODEL_NAME="$mname",HP_SEARCH="$HP_SEARCH",BATCH_SIZE="$BATCH_SIZE",PROMPT_TEMPLATE="$ptemp" \
            exp1_job.sh
            
        done
      done
    done
  done
done