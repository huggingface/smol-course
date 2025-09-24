# LoRA (Low-Rank Adaptation)

LoRA has become the most widely adopted PEFT method. It works by adding small rank decomposition matrices to the attention weights, typically reducing trainable parameters by about 90%. 

## Understanding LoRA

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that freezes the pre-trained model weights and injects trainable rank decomposition matrices into the model's layers. Instead of training all model parameters during fine-tuning, LoRA decomposes the weight updates into smaller matrices through low-rank decomposition, significantly reducing the number of trainable parameters while maintaining model performance. For example, when applied to GPT-3 175B, LoRA reduced trainable parameters by 10,000x and GPU memory requirements by 3x compared to full fine-tuning. You can read more about LoRA in the [LoRA paper](https://arxiv.org/pdf/2106.09685).

LoRA works by adding pairs of rank decomposition matrices to transformer layers, typically focusing on attention weights. During inference, these adapter weights can be merged with the base model, resulting in no additional latency overhead. LoRA is particularly useful for adapting large language models to specific tasks or domains while keeping resource requirements manageable.

## Loading LoRA Adapters

Adapters can be loaded onto a pretrained model with load_adapter(), which is useful for trying out different adapters whose weights aren’t merged. Set the active adapter weights with the set_adapter() function. To return the base model, you could use unload() to unload all of the LoRA modules. This makes it easy to switch between different task-specific weights.

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("<base_model_name>")
peft_model_id = "<peft_adapter_id>"
model = PeftModel.from_pretrained(base_model, peft_model_id)
```

![lora_load_adapter](./images/lora_adapter.png)

## Merging LoRA Adapters

After training with LoRA, you might want to merge the adapter weights back into the base model for easier deployment. This creates a single model with the combined weights, eliminating the need to load adapters separately during inference.

The merging process requires attention to memory management and precision. Since you'll need to load both the base model and adapter weights simultaneously, ensure sufficient GPU/CPU memory is available. Using `device_map="auto"` in `transformers` will help with automatic memory management. Maintain consistent precision (e.g., float16) throughout the process, matching the precision used during training and saving the merged model in the same format for deployment. Before deploying, always validate the merged model by comparing its outputs and performance metrics with the adapter-based version.

Adapters are also be convenient for switching between different tasks or domains. You can load the base model and adapter weights separately. This allows for quick switching between different task-specific weights. 

## Implementation Guide

The `notebooks/` directory contains practical tutorials and exercises for implementing different PEFT methods. Begin with `load_lora_adapter_example.ipynb` for a basic introduction, then explore `lora_finetuning.ipynb` for a more detailed look at how to fine-tune a model with LoRA and SFT.

When implementing PEFT methods, start with small rank values (4-8) for LoRA and monitor training loss. Use validation sets to prevent overfitting and compare results with full fine-tuning baselines when possible. The effectiveness of different methods can vary by task, so experimentation is key.

## OLoRA

[OLoRA](https://arxiv.org/abs/2406.01775) utilizes QR decomposition to initialize the LoRA adapters. OLoRA translates the base weights of the model by a factor of their QR decompositions, i.e., it mutates the weights before performing any training on them. This approach significantly improves stability, accelerates convergence speed, and ultimately achieves superior performance.

## Using TRL with PEFT

PEFT methods can be combined with TRL (Transformers Reinforcement Learning) for efficient fine-tuning. This integration is particularly useful for RLHF (Reinforcement Learning from Human Feedback) as it reduces memory requirements.

```python
from peft import LoraConfig
from transformers import AutoModelForCausalLM

# Load model with PEFT config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Load model on specific device
model = AutoModelForCausalLM.from_pretrained(
    "your-model-name",
    load_in_8bit=True,  # Optional: use 8-bit precision
    device_map="auto",
    peft_config=lora_config
)
```

Above, we used `device_map="auto"` to automatically assign the model to the correct device. You can also manually assign the model to a specific device using `device_map={"": device_index}`. You could also scale training across multiple GPUs while keeping memory usage efficient.

## Basic Merging Implementation

After training a LoRA adapter, you can merge the adapter weights back into the base model. Here's how to do it:

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 1. Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    "base_model_name",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. Load the PEFT model with adapter
peft_model = PeftModel.from_pretrained(
    base_model,
    "path/to/adapter",
    torch_dtype=torch.float16
)

# 3. Merge adapter weights with base model
try:
    merged_model = peft_model.merge_and_unload()
except RuntimeError as e:
    print(f"Merging failed: {e}")
    # Implement fallback strategy or memory optimization

# 4. Save the merged model
merged_model.save_pretrained("path/to/save/merged_model")
```

If you encounter size discrepancies in the saved model, ensure you're also saving the tokenizer:

```python
# Save both model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("base_model_name")
merged_model.save_pretrained("path/to/save/merged_model")
tokenizer.save_pretrained("path/to/save/merged_model")
```

## Next Steps

⏩ Move on to the [Prompt Tuning](prompt_tuning.md) guide to learn how to fine-tune a model with prompt tuning.
⏩ Move on the [Load LoRA Adapters Tutorial](./notebooks/load_lora_adapter.ipynb) to learn how to load LoRA adapters.

# Resources

- [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Hugging Face blog post on PEFT](https://huggingface.co/blog/peft)
