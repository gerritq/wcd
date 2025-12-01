generate_name() {
    
    local main="$1"
    local model_type="$2"
    local model_name="$3"
    local atl="$4"

    if [[ "$main" == "1" ]]; then
        MODEL_DIR="/scratch/prj/inf_nlg_ai_detection/wcd/data/exp1"

        if [[ "$model_type" == "slm" ]]; then
           if [[ "$atl" == "1" || "$atl" == "true" ]]; then
                name="atl_${training_size}"
            else
                name="van_${training_size}"
        fi
    else
        name="cls_${training_size}"
    f
    
    else 
        MODEL_DIR="/scratch/prj/inf_nlg_ai_detection/wcd/data/exp1_test"
    local name=""

    if [[ "$model_type" == "slm" ]]; then
        if [[ "$atl" == "1" || "$atl" == "true" ]]; then
            name="atl_${training_size}"
        else
            name="van_${training_size}"
        fi
    else
        name="cls_${training_size}"
    fi

    echo "$name"
}