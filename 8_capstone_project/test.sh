export HF_CACHE="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HUB_CACHE="$HF_CACHE"
export HF_HOME="$HF_CACHE"

lighteval accelerate "pretrained=HuggingFaceTB/SmolLM2-135M-Instruct" "community|example|20|1" --custom-tasks "submitted_tasks/example.py" --output-dir "results" --override-batch-size 512