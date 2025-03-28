# 偏好对齐

本章将学习如何将语言模型的输出和人类偏好对齐。虽然有监督微调（SFT）已经将模型适配到特定的任务领域了，但偏好对齐（Prefenrece Alignment）将会迫使模型的输出更加匹配人类的期望、符合人类的价值观。

## 概览

典型的偏好对齐方法一般都包含这几个步骤：
1. 使用 SFT 将模型适配到特定的领域
2. 使用偏好对齐（如 RLHF 或 DPO 算法）进一步提升模型回答的质量

其它偏好对齐算法还包括 ORPO，这个算法将指令微调和偏好对齐结合进了一个单一步骤中。本章我们将重点学习 DPO 和 ORPO 算法。

如果你还想进一步学习相关对齐算法，你可以阅读[这篇博客](https://argilla.io/blog/mantisnlp-rlhf-part-8)。

### 1️⃣ 直接偏好优化（DPO）

直接偏好优化（Direct Preference Optimization），简称 DPO，直接使用偏好数据对模型进行参数更新。这简化了偏好对齐的过程。这个方法无需额外设置激励模型、无需复杂强化学习步骤，比基于人类反馈的强化学习（RLHF）更高效更稳定。本章中对应的学习资料在[这里](./dpo.md)。

### 2️⃣ 基于优势比的偏好优化（ORPO）

基于优势比的偏好优化（Odds Ratio Preference Optimization），简称 ORPO，是一种将指令微调和偏好对齐结合在一起的方法。通过在 token 层面定义一个优势比（Odds），并在优势比上使用负对数似然损失函数，ORPO 改变了传统的语言建模的目标函数。ORPO 训练步骤简单、无需 DPO 中的参考模型，计算效率也更高。该方法在多项评测基准上展现了优秀的效果，尤其在 AlpacaEval 超越了传统方法。本章中对应的学习资料在[这里](./orpo.md)。

## 实践练习

| 标题 | 简介 | 习题 | 链接 | Colab |
|-------|-------------|----------|------|-------|
| DPO 训练 | 学习用 DPO 训练模型 | 🐢 在 Anthropic HH-RLHF 数据集上训练模型<br>🐕 使用你自己的偏好数据集<br>🦁 对不同的偏好数据集和不同大小的模型进行实验 | [Notebook](./notebooks/dpo_finetuning_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/2_preference_alignment/notebooks/dpo_finetuning_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| ORPO 训练 | 学习用 ORPO 训练模型 | 🐢 用指令数据和偏好数据训练模型<br>🐕 对不同的损失权重进行实验<br>🦁 对比 ORPO 和 DPO 的结果 | [Notebook](./notebooks/orpo_finetuning_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/2_preference_alignment/notebooks/orpo_finetuning_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |


## 参考资料

- [TRL 官方文档](https://huggingface.co/docs/trl/index) - TRL 是一个基于 Transformers 的强化学习库，这里实现了包括 DPO 在内的各种对齐算法。
- [DPO 论文](https://arxiv.org/abs/2305.18290) - 该论文针对当时已有的 RLHF 方法，提出了新的对齐方法，可以直接使用偏好数据优化模型参数。
- [ORPO 论文](https://arxiv.org/abs/2403.07691) - ORPO 算法将指令微调和偏好优化和并进一个训练步骤中。
- [RLHF 相关博客](https://argilla.io/blog/mantisnlp-rlhf-part-8/) - 这篇博客介绍了包括 RLHF、DPO 在内的对齐算法，同时也介绍了具体实现方法。
- [DPO 相关博客](https://huggingface.co/blog/dpo-trl) - 介绍了使用 TRL 实现 DPO 的具体步骤，包括示例代码和其它最佳实践经验。
- [DPO 示例训练脚本](https://github.com/huggingface/trl/blob/main/examples/scripts/dpo.py) - 完整的基于 TRL 的 DPO 训练代码。
- [ORPO 示例训练脚本](https://github.com/huggingface/trl/blob/main/examples/scripts/orpo.py) - 完整的基于 TRL 的 ORPO 训练代码。
- [Hugging Face 关于对齐训练的资料](https://github.com/huggingface/alignment-handbook) - 包括 SFT、DPO、RLHF 的语言模型对齐算法介绍，包括理论指导和实践代码。