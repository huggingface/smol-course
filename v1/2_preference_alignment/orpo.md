# Odds Ratio Preference Optimization (ORPO)

ORPO (Odds Ratio Preference Optimization) is a novel fine-tuning technique that combines fine-tuning and preference alignment into a single unified process. This combined approach offers advantages in efficiency and performance compared to traditional methods like RLHF or DPO.

## Understanding ORPO

Alignment with methods like DPO typically involve two separate steps: supervised fine-tuning to adapt the model to a domain and format, followed by preference alignment to align with human preferences. While SFT effectively adapts models to target domains, it can inadvertently increase the probability of generating both desirable and undesirable responses. ORPO addresses this limitation by integrating both steps into a single process, as illustrated in the comparison below:

![Alignment Techniques Comparison](https://argilla.io/images/blog/mantisnlp-rlhf/part-8-alignments.png)
*Comparison of different model alignment techniques*

## How ORPO Works

The training process leverages a preference dataset similar to what we used for DPO, where each training example contains an input prompt along with two responses: one that is preferred, and another that is rejected. Unlike other alignment methods that require separate stages and reference models, ORPO integrates preference alignment directly into the supervised fine-tuning process. This monolithic approach makes it reference model-free, computationally more efficient, and memory efficient with fewer FLOPs.

ORPO creates a new objective by combining two main components:

1. **SFT Loss**: The standard negative log-likelihood loss used in language modeling, which maximizes the probability of generating reference tokens. This helps maintain the model's general language capabilities.

2. **Odds Ratio Loss**: A novel component that penalizes undesirable responses while rewarding preferred ones. This loss function uses odds ratios to effectively contrast between favored and disfavored responses at the token level.

Together, these components guide the model to adapt to desired generations for the specific domain while actively discouraging generations from the set of rejected responses. The odds ratio mechanism provides a natural way to measure and optimize the model's preference between chosen and rejected outputs. If you want to deep dive into the math, you can read the [ORPO paper](https://arxiv.org/abs/2403.07691). If you want to learn about ORPO from the implementation perspective, you should check out how loss for ORPO is calculated in the [TRL library](https://github.com/huggingface/trl/blob/b02189aaa538f3a95f6abb0ab46c0a971bfde57e/trl/trainer/orpo_trainer.py#L660).

## Performance and Results

ORPO has demonstrated impressive results across various benchmarks. On MT-Bench, it achieves competitive scores across different categories:

![MT-Bench Results](https://argilla.io/images/blog/mantisnlp-rlhf/part-8-mtbench.png)
*MT-Bench results by category for Mistral-ORPO models*

When compared to other alignment methods, ORPO shows superior performance on AlpacaEval 2.0:

![AlpacaEval Results](https://argilla.io/images/blog/mantisnlp-rlhf/part-8-winrate.png)
*AlpacaEval 2.0 scores across different alignment methods*

Compared to SFT+DPO, ORPO reduces computational requirements by eliminating the need for a reference model and halving the number of forward passes per batch. Also, the training process is more stable across different model sizes and datasets, requiring fewer hyperparameters to tune. Performance-wise, ORPO matches larger models while showing better alignment with human preferences.

## Implementation 

Successful implementation of ORPO depends heavily on high-quality preference data. The training data should follow clear annotation guidelines and provide a balanced representation of preferred and rejected responses across diverse scenarios. 

### Implementation with TRL

ORPO can be implemented using the Transformers Reinforcement Learning (TRL) library. Here's a basic example:

```python
from trl import ORPOConfig, ORPOTrainer

# Configure ORPO training
orpo_config = ORPOConfig(
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_steps=1000,
    orpo_alpha=1.0,  # Controls strength of preference optimization
    orpo_beta=0.1,   # Temperature parameter for odds ratio
)

# Initialize trainer
trainer = ORPOTrainer(
    model=model,
    args=orpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# Start training
trainer.train()
```

Key parameters to consider:
- `orpo_alpha`: Controls the strength of preference optimization
- `orpo_beta`: Temperature parameter for the odds ratio calculation
- `learning_rate`: Should be relatively small to prevent catastrophic forgetting
- `gradient_accumulation_steps`: Helps with training stability

## Next Steps

⏩ Try the [ORPO Tutorial](./notebooks/orpo_finetuning_example.ipynb) to implement this unified approach to preference alignment.

## Resources
- [ORPO Paper](https://arxiv.org/abs/2403.07691)
- [TRL Documentation](https://huggingface.co/docs/trl/index)
- [Argilla RLHF Guide](https://argilla.io/blog/mantisnlp-rlhf-part-8/) 