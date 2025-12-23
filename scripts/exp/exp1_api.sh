#!/bin/bash
#SBATCH --job-name=exp1-api
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=cpu,nmes_cpu
#SBATCH --mem=1GB

LANGUAGE_LST=("en" "pt" "de" "ru" "it" "vi" "tr" "nl" "uk" "ro" "id" "bg" "uz" "no" "az" "mk" "hy" "sq")
MODEL_LST=("google/gemini-2.5-flash" "openai/gpt-5.1") # openai/gpt-5.1 google/gemini-2.5-flash
SHOTS_LST=(0 1) # 0 1
CONTEXT_LST=(1)
VERBOSE_LST=(0 1)
SMOKE_TEST=0

MODEL_DIR="/scratch/prj/inf_nlg_ai_detection/wcd/data/exp1"


for LANGUAGE in "${LANGUAGE_LST[@]}"; do
  for MODEL in "${MODEL_LST[@]}"; do
    for SHOT in "${SHOTS_LST[@]}"; do
      for CONTEXT in "${CONTEXT_LST[@]}"; do
        for VERBOSE in "${VERBOSE_LST[@]}"; do

            # Create a unique run dir
            if [[ "$SMOKE_TEST" -eq 0 ]]; then
                MODEL_LANG_DIR="${MODEL_DIR}/${LANGUAGE}"
                mkdir -p "$MODEL_LANG_DIR"
                RUN_DIR=$(mktemp -d "${MODEL_LANG_DIR}/run_XXXXXX")
            else
                RUN_DIR=""
            fi  


            echo "Running: lang=$LANGUAGE model=$MODEL shots=$SHOT ctx=$CONTEXT verbose=$VERBOSE"

            uv run utils/api.py \
                --lang "$LANGUAGE" \
                --model "$MODEL" \
                --shots "$SHOT" \
                --context "$CONTEXT" \
                --verbose "$VERBOSE" \
                --run_dir "$RUN_DIR" \
                --smoke_test "$SMOKE_TEST"

        done
      done
    done
  done
done