# 指令微调

本章内容主要聚焦大语言模型指令微调部分。指令微调会把预训练模型在特定领域的数据集上进一步训练，来适配特定的任务。这一过程能有效地提升模型在目标任务上的性能。

具体而言，本章将会重点探索两个主题：聊天模板和有监督微调。

## 1️⃣ 聊天模板

聊天模板的主要作用是把用户和 AI 模型之间的交互信息结构化，确保模型能够稳定输出且根据上下文作出回答。一个聊天模板包含系统提示词和人机两个角色发送的消息。本章的[聊天模板教程](./chat_templates.md)将会详细讲述这一内容。

## 2️⃣ 有监督微调

有监督微调（SFT）是你将预训练模型往特定任务迁移时的一个重要过程。SFT 通过在特定领域的有标注数据集上进一步训练，来提升模型在这里应用领域的性能。本章[有监督微调教程](./supervised_fine_tuning.md)将会详细讲解相关内容，包括其中的重要步骤和最佳实践。

## 实践练习

| 标题 | 简介 | 习题 | 链接 | Colab |
|-------|-------------|----------|------|-------|
| 聊天模板 | 学习使用 SmolLM2 的聊天模板，并将数据集转换为 ChatML 聊天模板的格式 | 🐢 将 `HuggingFaceTB/smoltalk` 数据集转换为 ChatnML 格式 <br> 🐕 将 `openai/gsm8k` 转换为 ChatML 格式 | [Notebook](./notebooks/chat_templates_example_cn.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/1_instruction_tuning/notebooks/chat_templates_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| 有监督微调 | 学习用 SFTTrainer 去微调 SmolLM2 模型 | 🐢 使用 `HuggingFaceTB/smoltalk` 数据集训练模型<br>🐕 使用 `bigcode/the-stack-smol` 数据集训练模型<br>🦁 针对一个实际场景选取数据集去训练 | [Notebook](./notebooks/sft_finetuning_example_cn.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/1_instruction_tuning/notebooks/sft_finetuning_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

## 参考资料

- [transformers 文档中关于聊天模板的介绍](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [使用 TRL 进行有监督微调的示例脚本](https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py)
- [TRL 官方文档关于 `SFTTrainer` 的介绍](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [DPO 算法论文](https://arxiv.org/abs/2305.18290)
- [TRL 官方文档中关于有监督微调的教程](https://huggingface.co/docs/trl/main/en/tutorials/supervised_finetuning)
- [博客：用 ChatML 和 TRL 微调 Google Gemma 模型](https://www.philschmid.de/fine-tune-google-gemma)
- [教程：微调大语言模型使其输出 JSON 格式的内容](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_to_generate_persian_product_catalogs_in_json_format)
