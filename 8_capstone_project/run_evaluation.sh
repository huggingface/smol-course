#!/bin/bash
export HF_CACHE="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HUB_CACHE="$HF_CACHE"
export HF_HOME="$HF_CACHE"

# Default values
num_fewshots=0
truncate_fewshots=0

# Function to print usage
print_usage() {
    echo "Usage: $0 -m MODEL_ID [-f NUM_FEWSHOTS] [-x]"
    echo "  -m MODEL_ID         : HuggingFace model ID (required)"
    echo "  -f NUM_FEWSHOTS     : Number of few-shot examples (default: 0)"
    echo "  -x                  : Truncate few-shot examples (default: false)"
    echo "  -h                  : Show this help message"
}

# Parse command line arguments
while getopts "m:f:xh" opt; do
    case $opt in
        m) model_id="$OPTARG";;
        f) num_fewshots="$OPTARG";;
        x) truncate_fewshots="true";;
        h) print_usage; exit 0;;
        ?) print_usage; exit 1;;
    esac
done

# Check if model_id is provided
if [ -z "$model_id" ]; then
    echo "Error: Model ID is required"
    print_usage
    exit 1
fi

# Get the directory of the script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tasks_dir="$script_dir/submitted_tasks"

# Create output directory if it doesn't exist
output_dir="$script_dir/results/$model_id"
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

# Collect all Python files (excluding run_evaluation.py)
task_files=()
task_names=()
while IFS= read -r -d '' file; do
    task_files+=("${file}")
    task_names+=($(basename "$file" .py))
done < <(find "$tasks_dir" -name "*.py" -print0)

# Check if any task files were found
if [ ${#task_files[@]} -eq 0 ]; then
    echo "Error: No Python task files found in $tasks_dir"
    exit 1
fi

echo "----------------------------------------"
echo "Running evaluation for model: $model_id"
echo "Found tasks: ${task_names[*]}"
echo "Number of few-shots: $num_fewshots"
echo "Truncate few-shots: $truncate_fewshots"

# Build the tasks parameter string
tasks_param=""
for task_name in "${task_names[@]}"; do
    tasks_param+="community|${task_name}|$num_fewshots|$truncate_fewshots,"
done
tasks_param=${tasks_param%,}  # Remove trailing comma

# Build the custom-tasks parameter string
custom_tasks_param=$(IFS=,; echo "${task_files[*]}")

lighteval accelerate "pretrained=$model_id" \
    "$tasks_param" \
    --custom-tasks "$custom_tasks_param" \
    --output-dir . \
    --override-batch-size 512

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "Successfully completed evaluation of all tasks"
    echo "Results saved in: $output_dir"
else
    echo "Error running evaluation (exit code: $exit_code)"
    exit $exit_code
fi 