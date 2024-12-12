# Prompt Tuning

Prompt tuning is a parameter-efficient approach that modifies input representations rather than model weights. Unlike traditional fine-tuning that updates all model parameters, prompt tuning adds and optimizes a small set of trainable tokens while keeping the base model frozen.

## Understanding Prompt Tuning

Prompt tuning is a parameter-efficient alternative to model fine-tuning that prepends trainable continuous vectors (soft prompts) to the input text. Unlike discrete text prompts, these soft prompts are learned through backpropagation while keeping the language model frozen. The method was introduced in ["The Power of Scale for Parameter-Efficient Prompt Tuning"](https://arxiv.org/abs/2104.08691) (Lester et al., 2021), which demonstrated that prompt tuning becomes more competitive with model fine-tuning as model size increases. Within the paper, at around 10 billion parameters, prompt tuning matches the performance of model fine-tuning while only modifying a few hundred parameters per task.

These soft prompts are continuous vectors in the model's embedding space that get optimized during training. Unlike traditional discrete prompts that use natural language tokens, soft prompts have no inherent meaning but learn to elicit the desired behavior from the frozen model through gradient descent. The technique is particularly effective for multi-task scenarios since each task requires storing only a small prompt vector (typically a few hundred parameters) rather than a full model copy. This approach not only maintains a minimal memory footprint but also enables rapid task switching by simply swapping prompt vectors without any model reloading.

## Training Process

Soft prompts typically number between 8 and 32 tokens and can be initialized either randomly or from existing text. The initialization method plays a crucial role in the training process, with text-based initialization often performing better than random initialization.

During training, only the prompt parameters are updated while the base model remains frozen. This focused approach uses standard training objectives but requires careful attention to the learning rate and gradient behavior of the prompt tokens.

## Implementation with PEFT

The PEFT library makes implementing prompt tuning straightforward. Here's a basic example:

```python
from peft import PromptTuningConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
model = AutoModelForCausalLM.from_pretrained("your-base-model")
tokenizer = AutoTokenizer.from_pretrained("your-base-model")

# Configure prompt tuning
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=8,  # Number of trainable tokens
    prompt_tuning_init="TEXT",  # Initialize from text
    prompt_tuning_init_text="Classify if this text is positive or negative:",
    tokenizer_name_or_path="your-base-model",
)

# Create prompt-tunable model
model = get_peft_model(model, peft_config)
```

## Comparison to Other Methods

When compared to other PEFT approaches, prompt tuning stands out for its efficiency. While LoRA offers low parameter counts and memory usage but requires loading adapters for task switching, prompt tuning achieves even lower resource usage and enables immediate task switching. Full fine-tuning, in contrast, demands significant resources and requires separate model copies for different tasks.

| Method | Parameters | Memory | Task Switching |
|--------|------------|---------|----------------|
| Prompt Tuning | Very Low | Minimal | Easy |
| LoRA | Low | Low | Requires Loading |
| Full Fine-tuning | High | High | New Model Copy |

When implementing prompt tuning, start with a small number of virtual tokens (8-16) and increase only if the task complexity demands it. Text initialization typically yields better results than random initialization, especially when using task-relevant text. The initialization strategy should reflect the complexity of your target task.

Training requires slightly different considerations than full fine-tuning. Higher learning rates often work well, but careful monitoring of prompt token gradients is essential. Regular validation on diverse examples helps ensure robust performance across different scenarios.

## Application

Prompt tuning excels in several scenarios:

1. Multi-task deployment
2. Resource-constrained environments
3. Rapid task adaptation
4. Privacy-sensitive applications

As models get smaller, prompt tuning becomes less competitive compared to full fine-tuning. For example, on models like SmolLM2 scales prompt tuning is less relevant than full fine-tuning. 

## Next Steps

⏭️ Move on to the [LoRA Adapters Tutorial](./notebooks/finetune_sft_peft.ipynb) to learn how to fine-tune a model with LoRA adapters.

## Resources
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Prompt Tuning Paper](https://arxiv.org/abs/2104.08691)
- [Hugging Face Cookbook](https://huggingface.co/learn/cookbook/prompt_tuning_peft)
