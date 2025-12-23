export MODEL_TYPE="clf"
export MODEL_NAME="llama3_8b"
export ATL=0
export SEEDS="42"

# ALL: "uk ro id bg uz no az mk hy sq"
export TARGET_LANGS="uk"
export SOURCE_LANGS="en"
export LANG="en"

export LANG_SETTINGS="main"
export CL_SETTINGS="zero"

export EXPERIMENT="cl_eval"

export LOWER_LR="1"

echo "Running exp3_job.sh with:"
env | grep -E 'LANG=|MODEL_TYPES=|MODEL_NAMES=|ATL=|SEEDS=|TARGET_LANGS=|SOURCE_LANGS=|LANG_SETTINGS=|CL_SETTINGS=|EXPERIMENT='
echo

# ---- run the job script ----
bash exp2_job.sh