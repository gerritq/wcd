#!/bin/bash
set -euo pipefail


# TEST_LANGS=("no")
# TRAINING_LANG_SETS=(
#   "en"
#   "en+nl"
# )

# TEST_LANGS=("bg")
# TRAINING_LANG_SETS=(
#   "en"
#   "en+ru"
#   "ru"
# )

# TEST_LANGS=("ro")
# TRAINING_LANG_SETS=(
#   "en"
#   "en+it"
#   "it"
# )

MODEL_TYPES=("slm")
MODEL_NAME="llama3_8b"
ATL_VALUES=(1)
TRAINING_SIZES=(5000) # this is the size per LANG!!!!!!

TIME="03:00:00"

for test_lang in "${TEST_LANGS[@]}"; do
  for training_langs in "${TRAINING_LANG_SETS[@]}"; do
    for model_type in "${MODEL_TYPES[@]}"; do
      for atl in "${ATL_VALUES[@]}"; do
        for training_size in "${TRAINING_SIZES[@]}"; do

          # skip classifier + atl=1 since it has no effect
          if [[ "$model_type" == "clf" && "$atl" -eq 1 ]]; then
            echo "Skipping classifier with ATL=1"
            continue
          fi

          job_name="e2-${model_type}-${test_lang}-tl[${training_langs// /_}]-atl${atl}-ts${training_size}"
          echo "$job_name"
        
          sbatch \
            --job-name="$job_name" \
            --time="$TIME" \
            --export=ALL,TEST_LANG="$test_lang",TRAINING_LANGS="$training_langs",MODEL_TYPE="$model_type",ATL="$atl",MODEL_NAME="$MODEL_NAME",TRAINING_SIZE="$training_size" \
            exp2_job.sh

        done
      done
    done
  done
done