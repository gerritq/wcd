export MODEL_TYPE="plm"
export MODEL_NAME="mBert"
export ATL=0
export SEEDS="42"

export TARGET_LANGS="uk no"
export SOURCE_LANGS="en"

export LANG_SETTINGS="main translation"
export CL_SETTINGS="zero"

export EXPERIMENT="cl_eval"

echo "Running exp3_job.sh with:"
env | grep -E 'MODEL_TYPES=|MODEL_NAMES=|ATL=|SEEDS=|TARGET_LANGS=|SOURCE_LANGS=|LANG_SETTINGS=|CL_SETTINGS=|EXPERIMENT='
echo

# ---- run the job script ----
bash exp3_job.sh