# 直接偏好优化（DPO）

直接偏好优化（Direct Preference Optimization），简称 DPO，是一种非常简洁的使用人类偏好数据对齐模型的算法。DPO 直接使用偏好数据优化模型，无需 RLHF 的激励模型和强化学习步骤。

## 理解 DPO

DPO 将偏好对齐任务转化为了一个在偏好数据上训练的分类任务。传统的 RLHF 需要训练一个额外的激励模型，并使用强化学习方法（如 PPO）去对齐模型输出。DPO简化了这个步骤，通过定义一个损失函数，直接在“倾向的输出”和“不倾向的输出”上进行训练。

这个方法在实践中十分高效，Llama 模型的训练就使用了 DPO。同时，没有了激励模型了强化学习，DPO 训练也更简单、更稳定。

## DPO 工作原理

在 DPO 之前，我们需要使用 SFT 微调模型，用指令跟随的数据集先把模型适配到特定任务领域中，让模型在这个领域具备基本能力。

接下来才是偏好学习。模型将在“倾向的输出”和“不倾向的输出”这样成对的数据上训练，学习哪种类型的回答更符合人类的喜好。

DPO 的关键原理在于它直接使用偏好数据进行优化。不同于 RLHF，DPO 使用了二分类的交叉墒损失函数，这里的损失直接在“倾向的输出”和“不倾向的输出”这样的成对数据上计算。这使得模型训练更稳定、更高效，同时效果甚至还比 RLHF 好。

## DPO 数据集

构造 DPO 专用数据集，一般需要对回答进行“倾向”和“不倾向”的标注。使用人工标注或自动化方法都可以实现这一步骤。下表就是一个示例数据集：

| Prompt | Chosen | Rejected |
|--------|--------|----------|
| ...    | ...    | ...      |
| ...    | ...    | ...      |
| ...    | ...    | ...      |

`Prompt` 这一栏提供问题， `Chosen` 和 `Rejected` 分别代表针对这个问题我们倾向的回答和不倾向的回答。`chosen` 和 `rejected` 也可以是一个列表形式，包含多个不同的回答。

你可以在 Hugging Face 的[这个地方](https://huggingface.co/collections/argilla/preference-datasets-for-dpo-656f0ce6a00ad2dc33069478)找到很多 DPO 数据集。

## 用 TRL 实现 DPO

使用 TRL 实现 DPO 非常简单直接，仅需配置 `DPOConfig` 和 `DPOTrainer` 即可。这两个类遵循 `transformers` 的 API 风格。
下面就是一个简单的例子：

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

我们还将在 [DPO 教程](./notebooks/dpo_finetuning_example.ipynb) 中详细讲解 `DPOConfig` 和 `DPOTrainer` 的配置。

## 最佳实践

数据质量对 DPO 的成败至关重要。偏好数据集必须足够多样，涵盖不同的想要的回答。在数据标注过程中，需要制定清晰明确的标注指导。通过提高数据集质量一般都可以提升模型性能，可能的做法包括对大规模数据集进行过滤，仅保留高质量数据，或仅保留和应用领域相关的数据。

训练过程中，仔细监视损失的收敛情况、及时验证性能也很重要。及时调节 $\beta$ 参数，在偏好学习和通用能力间找到平衡。有规律地在多样的问题上做验证测试，确保模型不过你和。这些也都很重要。

同时，也要对比一下原模型和优化后模型针对同一问题的回答，看看 模型是否学到了偏好。在包括极端情况下的问题集上测试，确保模型健壮性。

## 接下来的学习

⏩ 在 [DPO 教程](./notebooks/dpo_finetuning_example.ipynb)中，你可以直接上手实践。该教程将会带你实践 DPO 的整个过程，从数据准备指导模型训练和验证。

⏭️ 之后，你还可以学习 [ORPO](./orpo.md)，了解更多偏好优化算法。