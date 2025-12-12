#!/bin/bash
set -euo pipefail


# --------------------------------------------
# High to high resource
# --------------------------------------------

# TEST_LANGS=("vi")
# TRAINING_LANG_SETS=(
#   "en"
# )

# TEST_LANGS=("ru")
# TRAINING_LANG_SETS=(
#   "en"
# )


# TEST_LANGS=("de")
# TRAINING_LANG_SETS=(
#   "en"
# )

# TEST_LANGS=("nl")
# TRAINING_LANG_SETS=(
#   "en"
# )


# --------------------------------------------
# High to mid resource
# --------------------------------------------
# TEST_LANGS=("uk")
# TRAINING_LANG_SETS=(
#   "en"
#   "ru"
#   "en+ru"
# )

# TEST_LANGS=("uz")
# TRAINING_LANG_SETS=(
#   "en"
#   "tr"
#   "en+tr"
# )

# TEST_LANGS=("ro")
# TRAINING_LANG_SETS=(
#   "en"
#   "it"
#   "en+it"
# )

# TEST_LANGS=("bg")
# TRAINING_LANG_SETS=(
#   "en"
#   "ru"
#   "en+ru"
# )

# TEST_LANGS=("id")
# TRAINING_LANG_SETS=(
#   "en"
#   "vi"
# )

# --------------------------------------------
# High to low-resource
# --------------------------------------------

# TEST_LANGS=("no")
# TRAINING_LANG_SETS=(
#   "en"
#   "de"
#   "en+de"
# )

# TEST_LANGS=("az")
# TRAINING_LANG_SETS=(
#   # "en"
#   # "tr"
#   "en+tr"
# )

# TEST_LANGS=("mk")
# TRAINING_LANG_SETS=(
#   "en"
#   "ru"
#   "en+ru"
# )

# TEST_LANGS=("hy")
# TRAINING_LANG_SETS=(
#   "en"
# )

# TEST_LANGS=("sq")
# TRAINING_LANG_SETS=(
#   "en"
# )

MODEL_TYPES=("plm")
MODEL_NAME="llama3_8b" # "xlm-r-b"
ATL_VALUES=(1)
TRAINING_SIZES=(5000) # this is the size per LANG!!!!!!
HP_SEARCH=1

TIME="01:00:00"

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
            --export=ALL,TEST_LANG="$test_lang",TRAINING_LANGS="$training_langs",MODEL_TYPE="$model_type",ATL="$atl",MODEL_NAME="$MODEL_NAME",HP_SEARCH="$HP_SEARCH",TRAINING_SIZE="$training_size" \
            exp2_job.sh

        done
      done
    done
  done
done