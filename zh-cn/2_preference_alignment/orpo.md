# 基于优势比的偏好优化（ORPO）

基于优势比的偏好优化（Odds Ratio Preference Optimization）或 ORPO，是一种更新颖的偏好对齐方法，它把微调和偏好对齐结合，组成一个统一的过程。这个算法相比于 RLHF 和 DPO 有着更高的效率和更好的性能。

## 理解 ORPO

诸如 DPO 的对齐方法一般包含两个步骤：使用 SFT 先让模型适配如这个领域或回答格式，然后进行偏好对齐训练。虽然 SFT 已经将模型对齐到了特定任务领域，但模型不可避免可能会产生我们不期望的回答，所以我们还需要进行下一步的偏好对齐。ORPO 则合并了这两个步骤。下图取自 ORPO 论文，对比了 RLHF、DPO 和 ORPO 的差异：

![Alignment Techniques Comparison](https://argilla.io/images/blog/mantisnlp-rlhf/part-8-alignments.png)
*三种对齐算法的对比*

## ORPO 工作原理

ORPO 训练使用的数据集和 DPO 相似：针对一个输入的问题包含两个可能输出：一个是“倾向的输出”，另一个是“不倾向的输出”。不同的是，ORPO 直接将偏好对齐加入到 SFT 中。这一整体性方法使得它无需 DPO 中的参考模型，同时也更高效、节省内存。

ORPO 的损失函数包含两个部分：

1. **SFT Loss**：这里使用标准的负对数似然函数，和 DPO 中的类似，用于扩大想要的 token 的生成概率。这个损失函数也有助于模型保持通用的语言能力。
2. **Odds Ratio Loss**：这是新提出的损失函数。由于上述 SFT Loss 不能惩罚不想要的输出，所以这个函数在激励倾向的输出的同时，也惩罚不倾向的输出。具体来说，这里定义了计算优势比（Odds Ratio）的公式，通过抬高倾向输出和不倾向输出两者的优势比比值，在奖励倾向输出的同时，也压低不倾向输出的生成概率。

在两个损失函数共同作用下，模型不仅被适配进了相应的任务领域，也压低了不倾向的回答的生成概率。其中，优势比这个机制提供了一个很直观的方法，模拟了倾向回答和不倾向回答之间的差异程度。你也可以阅读 [ORPO 的论文](https://arxiv.org/abs/2403.07691)，进一步了解其中的数学理论。如果你对具体的实现感兴趣，你可以阅读 [TRL 中关于这部分的实现](https://github.com/huggingface/trl/blob/b02189aaa538f3a95f6abb0ab46c0a971bfde57e/trl/trainer/orpo_trainer.py#L660)。

## 训练结果

ORPO 在很多测试基准上都取得了不错的效果。以下是它在 MT-Bench 测试基准上的结果（根据任务类别划分）：

![MT-Bench Results](https://argilla.io/images/blog/mantisnlp-rlhf/part-8-mtbench.png)
*Mistral-ORPO 模型在 MT-Bench 不同任务领域的结果*

在 AlpacaEval 2.0 上，ORPO 展现了超越其它对齐算法的效果：

![AlpacaEval Results](https://argilla.io/images/blog/mantisnlp-rlhf/part-8-winrate.png)
*不同对齐算法在 AlpacaEval 2.0 的得分*

相较于 SFT 加 DPO 的做法，ORPO 通过去除参考模型、减半前向推理的策略，大大降低了计算资源的要求。同时，训练过程也更稳定，需要调节的超参数也更少。在性能上，ORPO 对人类偏好的适配做得也更好。

## 代码实现

成功训练 ORPO 也极度依赖高质量数据集。所以标注训练数据时，我们也需要清晰明确的标注标准，确保对话场景的多样性，同时倾向和不倾向的回答需要分布均匀。

### 用 TRL 实现 ORPO

以下代码提供了用 TRL 实现 ORPO 的基本示例：

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

关键参数：
- `orpo_alpha`：用来控制偏好优化部分的权重
- `orpo_beta`：计算优势比（Odds Ratio）时的 Temperature 参数
- `learning_rate`：这里需要用较小的学习率，用来防治灾难性遗忘（catastrophic forgetting）
- `gradient_accumulation_steps`：调节这个也能稳定训练

## 接下来的学习

⏩ 学习 [ORPO 教程](./notebooks/orpo_finetuning_example.ipynb) 来实践 ORPO 算法。

## 学习资源
- [ORPO 论文](https://arxiv.org/abs/2403.07691)
- [TRL 官方文档](https://huggingface.co/docs/trl/index)
- [Argilla RLHF Guide](https://argilla.io/blog/mantisnlp-rlhf-part-8/) 