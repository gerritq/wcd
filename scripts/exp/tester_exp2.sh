export MODEL_TYPE="plm"
export MODEL_NAME="mBert"
export ATL=0
export SEEDS="42"


export TARGET_LANGS="uk"
export SOURCE_LANGS="en"
export LANG="en"

export LANG_SETTINGS="main"
export CL_SETTINGS="few"

export EXPERIMENT="cl_eval"

echo "Running exp3_job.sh with:"
env | grep -E 'LANG=|MODEL_TYPES=|MODEL_NAMES=|ATL=|SEEDS=|TARGET_LANGS=|SOURCE_LANGS=|LANG_SETTINGS=|CL_SETTINGS=|EXPERIMENT='
echo

# ---- run the job script ----
bash exp2_job.sh