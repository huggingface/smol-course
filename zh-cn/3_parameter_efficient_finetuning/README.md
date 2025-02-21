# 高效的参数微调（PEFT）

随着语言模型越来越大，使用传统的模型微调方法去微调 LLM 已经变得越来越难了。举例来说，微调一个 1.7B 参数量的模型需要把所有参数都放进 GPU 显存、保存模型优化和状态信息，甚至还需要保存模型备份，这就需要很大的 GPU 显存使用了。同时，微调所有参数还有很大的“灾难性遗忘”风险，可能损失模型原有的能力。针对此问题，高效的参数微调（Parameter-efficient fine-tuning 或 PEFT）被提出。它在微调模型时，保留大部分参数不变，只微调一小部分参数，大大节省了计算资源的需求。

传统微调需要更新所有模型参数，对大模型很不现实。而 PEFT 相关方法则发现，仅更新一小部分参数，就足以对模型进行适配，达到微调所期待的效果。这部分需要更新的参数甚至还不到总参数量的 1%。这一重大改进使得以下操作成为可能：

- 在 GPU 显存受限的消费级显卡上微调 LLM
- 高效地为不同任务领域保存不同的微调结果
- 在数据量不足的微调场景下也可保持很好的泛化性
- 微调训练耗时更少
  
## PEFT 相关算法

在本章教程中，我们主要讲解两种比较常用的 PEFT 算法：

### 1️⃣ LoRA (Low-Rank Adaptation)

LoRA 可以说是最常用的 PEFT 算法了，它为高效模型微调提供了一个非常优雅的解决方案。LoRA 在需要更新的参数（一般是 attention layers 的参数）上插入可以训练的参数矩阵，训练时仅训练这部分参数。当模型训练好后，我们会利用这部分参数对原有模型进行重参数化（re-parameterization）。这样可以不改变 LLM 的整体结构和参数数量。通过这种方法，需要更新的参数量能至少减少 90%，同时性能也不差于全量参数微调。我们将在 [LoRA 低秩分解](./lora_adapters_cn.md)部分进一步讲解。

### 2️⃣ Prompt Tuning

Prompt Tuning 则更加轻量化。它通过在输入部分加入**可训练的 token** 来微调，而不是改变模型的参数。Prompt Tuning 没有 LoRA 那么常用，但对于适配模型到新的任务领域来说，是个非常有用的技术。我们将在 [Prompt Tuning](./prompt_tuning_cn.md) 部分进一步讲解。

## 实践练习

| 标题 | 简介 | 习题 | 链接 | Colab |
|-------|-------------|----------|------|-------|
| LoRA 微调 | 学习使用 LoRA adapters 微调模型 | 🐢 用 LoRA 训练一个模型 <br>🐕 试验不同低秩值的效果 <br>🦁 与全量参数微调的效果进行对比 | [Notebook](./notebooks/finetune_sft_peft.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/3_parameter_efficient_finetuning/notebooks/finetune_sft_peft.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| 载入 LoRA Adapter | 学习如何加载并使用 LoRA adapters | 🐢 加载训练好的 adapter<br>🐕 将 adapter 融入原有模型中 <br>🦁 实现不同 adapter 的切换 | [Notebook](./notebooks/load_lora_adapter.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/3_parameter_efficient_finetuning/notebooks/load_lora_adapter.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
<!-- | Prompt Tuning | Learn how to implement prompt tuning | 🐢 Train soft prompts<br>🐕 Compare different initialization strategies<br>🦁 Evaluate on multiple tasks | [Notebook](./notebooks/prompt_tuning_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/3_parameter_efficient_finetuning/notebooks/prompt_tuning_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | -->

## 参考资料
- [PEFT 代码库官方文档](https://huggingface.co/docs/peft)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314)
- [Prompt Tuning 论文](https://arxiv.org/abs/2104.08691)
- [Hugging Face PEFT 代码库相关博客](https://huggingface.co/blog/peft)
- [文章：How to Fine-Tune LLMs in 2024 with Hugging Face](https://www.philschmid.de/fine-tune-llms-in-2024-with-trl) 
- [TRL 代码库官方文档](https://huggingface.co/docs/trl/index)
