# 视觉语言模型

## 1. 视觉语言模型的用处

视觉语言模型（Vision Language Models 或简称 VLM）是一种不仅仅接收语言输入、而且可以处理图片输入的模型，它可以支持诸如图像文本描述生成（image captioning）、视觉问答（visual question answering）、多模态推理（multimodal reasoning）等任务。

一个典型的 VLM 架构包含一个图像编码器（用来提取视觉特征）、一个映射层（用来对齐视觉特征和文本表征）以及一个语言模型（用以处理视觉语言特征并输出文本）。这使得模型得以在视觉元素和语言概念之间建立起连接。

VLM 可以有很多用处。基本的用途包括通用的视觉语言任务，而那些针对聊天对话场景优化过的 VLM 则可以支持对话式的人机互动。还有些模型可以根据图像中的证据预测事实，或者进行特定的任务，如物体检测。

关于 VLM 的更多技术和使用，建议读者在 [VLM 的使用](./vlm_usage_cn.md)这一节中学习。

## 2. 视觉语言模型的微调

微调一个 VLM 的过程，通常是指选择一个预训练过的模型，在一个新的数据集上学习处理特定领域的任务。这个过程可以参考本教程第 1 和 2 章的相关方法，比如有监督微调、偏好优化等。

虽然核心工具与技术和大语言模型（LLMs）所用的大致相同，但微调视觉语言模型（VLMs）需要格外关注图像的数据表示与准备。如此才能确保模型有效整合并处理视觉与文本数据，以实现最佳性能。鉴于演示模型 SmolVLM 比前一模块使用的语言模型大得多，探索高效的微调方法至关重要。量化和参数高效微调（PEFT）等技术，能让这一过程更易于操作且成本更低，使更多用户能够对该模型进行试验。

如果你想了解详细的微调 VLM 的技术，你可以学习 [VLM 微调](./vlm_finetuning_cn.md) 这一节。  


## Exercise Notebooks  


| 标题 | 概述 | 练习 | 链接 | Colab |
|-------|-------------|----------|------|-------|
| VLM 的使用 | 学习如何载入并使用一个预训练过的 VLM 来处理各种任务 | 🐢 尝试处理一张图片 <br>🐕 尝试用 batch 的方式处理多个图片 <br>🦁 尝试处理一整个视频 | [Notebook](./notebooks/vlm_usage_sample_cn.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/5_vision_language_models/notebooks/vlm_usage_sample.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| VLM 微调 | 学习针对特定任务数据集来微调一个预训练过的 VLM | 🐢 使用一个基本数据集进行微调<br>🐕 尝试使用新数据集 <br>🦁 试验一种不同的微调方法 | [Notebook](./notebooks/vlm_sft_sample_cn.ipynb)| <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/5_vision_language_models/notebooks/vlm_sft_sample.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | 


## 参考资料  
- [Hugging Face Learn: Supervised Fine-Tuning VLMs](https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl)
- [Hugging Face Learn: Supervised Fine-Tuning SmolVLM](https://huggingface.co/learn/cookbook/fine_tuning_smol_vlm_sft_trl)  
- [Hugging Face Learn: Preference Optimization Fine-Tuning SmolVLM](https://huggingface.co/learn/cookbook/fine_tuning_vlm_dpo_smolvlm_instruct)  
- [Hugging Face Blog: Preference Optimization for VLMs](https://huggingface.co/blog/dpo_vlm)
- [Hugging Face Blog: Vision Language Models](https://huggingface.co/blog/vlms)
- [Hugging Face Blog: SmolVLM](https://huggingface.co/blog/smolvlm)  
- [Hugging Face Model: SmolVLM-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)
- [CLIP: Learning Transferable Visual Models from Natural Language Supervision](https://arxiv.org/abs/2103.00020)  
- [Align Before Fuse: Vision and Language Representation Learning with Momentum Distillation](https://arxiv.org/abs/2107.07651)  
