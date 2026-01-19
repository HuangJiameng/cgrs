#!/bin/bash

declare -a MODEL_NAMES=( 
    "/path/to/model1"
    "/path/to/model2"
)

declare -a DATASETS=(
    "amc"
    "aime"
    "math"
    "gpqa"
)
DATASET_DIR="./data/"
OUTPUT_PATH="output" 
MAX_GENERATED_TOKENS=16384
POLICY="entropy" 
DELTA=0.9
DTYPE="bfloat16"
GPU_MEMORY_UTILIZATION=0.9
TRIAL_INDICES=(1 2 3) 
GPU_TOTAL_COUNT=8 
TEMPERATURE=0.6
current_combination_idx=0
wait_time=60

declare -a GPU_COMBINATIONS=(
    "0 1"
    "2 3"
    "4 5"
    "6 7"
)

check_gpu_pair_availability() {
    local gpu1_id=$1
    local gpu2_id=$2
    local memory_threshold=100

    echo "Checking availability for GPU pair ($gpu1_id, $gpu2_id)..."
    while true; do
        local gpu1_memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu1_id" 2>/dev/null)
        local gpu2_memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu2_id" 2>/dev/null)
        
        if [[ -z "$gpu1_memory_used" || ! "$gpu1_memory_used" =~ ^[0-9]+$ || \
              -z "$gpu2_memory_used" || ! "$gpu2_memory_used" =~ ^[0-9]+$ ]]; then
            echo "Warning: Unable to get valid memory usage for GPU pair ($gpu1_id, $gpu2_id). Retrying in $wait_time seconds."
            sleep $wait_time
            continue
        fi

        if (( gpu1_memory_used < memory_threshold && gpu2_memory_used < memory_threshold )); then 
            echo "GPU pair ($gpu1_id, $gpu2_id) is free (GPU $gpu1_id: ${gpu1_memory_used}MB, GPU $gpu2_id: ${gpu2_memory_used}MB <= ${memory_threshold}MB threshold)."
            sleep 10 # recheck for robustness
            if (( gpu1_memory_used < memory_threshold && gpu2_memory_used < memory_threshold )); then
                break 
            fi
        else
            echo "GPU pair ($gpu1_id, $gpu2_id) is in use (GPU $gpu1_id: ${gpu1_memory_used}MB, GPU $gpu2_id: ${gpu2_memory_used}MB). Waiting for $wait_time seconds..."
            
            sleep $wait_time
            current_combination_idx=$(( (current_combination_idx + 1) % ${#GPU_COMBINATIONS[@]} ))
            read -r next_gpu1 next_gpu2 <<< "${GPU_COMBINATIONS[current_combination_idx]}"
            gpu1_id=$next_gpu1
            gpu2_id=$next_gpu2
            echo "Moving to check next GPU pair: ($gpu1_id, $gpu2_id)"
        fi
    done
}

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        echo "Preparing to run tasks for Model: $MODEL_NAME, Dataset: $DATASET..."
        
        for i in "${!TRIAL_INDICES[@]}"; do
            TRIAL_INDEX=${TRIAL_INDICES[$i]}

            read -r gpu_id_main gpu_id_secondary <<< "${GPU_COMBINATIONS[current_combination_idx]}"

            check_gpu_pair_availability "$gpu_id_main" "$gpu_id_secondary" 

            read -r gpu_id_main gpu_id_secondary <<< "${GPU_COMBINATIONS[current_combination_idx]}"
            
            echo "Starting process for Model: $MODEL_NAME, Dataset: $DATASET, GPUs: ($gpu_id_main, $gpu_id_secondary), TRIAL_INDEX: $TRIAL_INDEX"

            CMD="CUDA_VISIBLE_DEVICES=$gpu_id_main,$gpu_id_secondary VLLM_USE_V1=0 python cgrs.py"
            CMD+=" --model_name_or_path \"$MODEL_NAME\""
            CMD+=" --dataset_dir \"$DATASET_DIR\""
            CMD+=" --output_path \"$OUTPUT_PATH\""
            CMD+=" --dataset \"$DATASET\""
            CMD+=" --max_generated_tokens $MAX_GENERATED_TOKENS"
            CMD+=" --policy $POLICY"
            CMD+=" --delta $DELTA"
            CMD+=" --dtype $DTYPE"
            CMD+=" --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"
            CMD+=" --trial_idx $TRIAL_INDEX  --temperature $TEMPERATURE" 

            eval "$CMD" &

            current_combination_idx=$(( (current_combination_idx + 1) % ${#GPU_COMBINATIONS[@]} ))
            
            sleep 15
        done
    done
done

echo "All tasks submitted."

