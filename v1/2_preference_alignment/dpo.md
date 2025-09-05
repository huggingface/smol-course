# Direct Preference Optimization (DPO)

Direct Preference Optimization (DPO) offers a simplified approach to aligning language models with human preferences. Unlike traditional RLHF methods that require separate reward models and complex reinforcement learning, DPO directly optimizes the model using preference data.

## Understanding DPO

DPO recasts preference alignment as a classification problem on human preference data. Traditional RLHF approaches require training a separate reward model and using complex reinforcement learning algorithms like PPO to align model outputs. DPO simplifies this process by defining a loss function that directly optimizes the model's policy based on preferred vs non-preferred outputs.

This approach has proven highly effective in practice, being used to train models like Llama. By eliminating the need for a separate reward model and reinforcement learning stage, DPO makes preference alignment more accessible and stable.

## How DPO Works

The DPO process requires supervised fine-tuning (SFT) to adapt the model to the target domain. This creates a foundation for preference learning by training on standard instruction-following datasets. The model learns basic task completion while maintaining its general capabilities.

Next comes preference learning, where the model is trained on pairs of outputs - one preferred and one non-preferred. The preference pairs help the model understand which responses better align with human values and expectations.

The core innovation of DPO lies in its direct optimization approach. Rather than training a separate reward model, DPO uses a binary cross-entropy loss to directly update the model weights based on preference data. This streamlined process makes training more stable and efficient while achieving comparable or better results than traditional RLHF.

## DPO datasets

Datasets for DPO are typically created by annotating pairs of responses as preferred or non-preferred. This can be done manually or using automated filtering techniques. Below is an example structure of single turn preference dataset for DPO:

| Prompt | Chosen | Rejected |
|--------|--------|----------|
| ...    | ...    | ...      |
| ...    | ...    | ...      |
| ...    | ...    | ...      |

The `Prompt` column contains the prompt used to generate the `Chosen` and `Rejected` responses. The `Chosen` and `Rejected` columns contain the responses that are preferred and non-preferred respectively. There are variations on this structure, for example, including a system prompt column or `Input` column containing reference material. The values of `chosen` and `rejected` can be be represented as strings for single turn conversations or as conversation lists. 

You can find a collection of DPO datasets on Hugging Face [here](https://huggingface.co/collections/argilla/preference-datasets-for-dpo-656f0ce6a00ad2dc33069478).

## Implementation with TRL

The Transformers Reinforcement Learning (TRL) library makes implementing DPO straightforward. The `DPOConfig` and `DPOTrainer` classes follow the same `transformers` style API.
Here's a basic example of setting up DPO training:

```python
from trl import DPOConfig, DPOTrainer

# Define arguments
training_args = DPOConfig(
    ...
)

# Initialize trainer
trainer = DPOTrainer(
    model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    ...
)

# Train model
trainer.train()
```

We will cover more details on how to use the `DPOConfig` and `DPOTrainer` classes in the [DPO Tutorial](./notebooks/dpo_finetuning_example.ipynb).

## Best Practices

Data quality is crucial for successful DPO implementation. The preference dataset should include diverse examples covering different aspects of desired behavior. Clear annotation guidelines ensure consistent labeling of preferred and non-preferred responses. You can improve model performance by improving the quality of your preference dataset. For example, by filtering down larger datasets to include only high quality examples, or examples that relate to your use case.

During training, carefully monitor the loss convergence and validate performance on held-out data. The beta parameter may need adjustment to balance preference learning with maintaining the model's general capabilities. Regular evaluation on diverse prompts helps ensure the model is learning the intended preferences without overfitting.

Compare the model's outputs with the reference model to verify improvement in preference alignment. Testing on a variety of prompts, including edge cases, helps ensure robust preference learning across different scenarios.

## Next Steps

⏩ To get hands-on experience with DPO, try the [DPO Tutorial](./notebooks/dpo_finetuning_example.ipynb). This practical guide will walk you through implementing preference alignment with your own model, from data preparation to training and evaluation. 

⏭️ After completing the tutorial, you can explore the [ORPO](./orpo.md) page to learn about another preference alignment technique.