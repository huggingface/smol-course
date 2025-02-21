# 评测

在开发和部署语言模型的过程中，模型评测是一个非常关键的步骤。它有助于我们理解模型在不同方面的能力究竟如何，并找到进一步提示改进的空间。本章将涵盖标准基准测试和特定领域的评估方法，以全面评估你的 smol 模型。

我们将使用 [`lighteval`](https://github.com/huggingface/lighteval) 这个强大的评测代码库。它由 HuggingFace 开发，并完美地集成入了 HuggingFace 的生态系统。我们还提供了[指南书籍](https://github.com/huggingface/evaluation-guidebook)以便读者想要深入学习评测的相关概念和最佳实践。

## 章节总览

一个全面的评估策略会检查模型多个方面的性能。我们将评估模型特定领域的能力，比如回答问题、概括总结，来理解模型处理不同问题的能力。我们通过生成的连贯性和事实准确性等因素来衡量输出质量。同时，我们也需要安全评测，防治模型输出有害的信息或带有偏见的观点。最后，我们还可以进行特定领域的专业性测试，来确认模型是否在特定领域掌握了专业知识。

## 目录

### 1️⃣ [自动化基准测试](./automatic_benchmarks_cn.md)

学习如何使用标准的测试基准和指标来评估模型。我们将会学习常见的测试基准，如 MMLU 和 TruthfulQA，理解重要指标和相关配置，同时为可复现的评估结果提供最佳实践。

### 2️⃣ [自定义领域的评测](./custom_evaluation_cn.md)

学习怎样为你的特定任务领域量身定做评估流程。我们将学习设计自定义评估任务、代码实现特定的指标，以及构建符合你要求的评估数据集。

### 3️⃣ [领域评估的项目示例](./project/README_CN.md)

通过一个完整的例子，学习构建特定领域的评测流程。这包含：生成评测数据集、使用 Argilla 平台进行数据标注、构建标准化的数据集、用 LightEval 评测模型。

### Notebook 练习

| 标题 | 简述 | 习题 | 链接 | Colab |
|-------|-------------|----------|------|-------|
| 评测并分析你的大语言模型 | 学习使用 LightEval 在特定领域评测、比较模型 | 🐢 使用医学相关领域的任务评估模型 <br> 🐕 Create a new domain evaluation with different MMLU tasks <br> 🦁 为你的特定任务领域创建一个自定义的评测任务 | [Notebook](./notebooks/lighteval_evaluate_and_analyse_your_LLM.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/4_evaluation/notebooks/lighteval_evaluate_and_analyse_your_LLM.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |


## Resources

- [评测的指南书籍](https://github.com/huggingface/evaluation-guidebook) - 大语言模型评测领域的全面指南
- [LightEval 文档](https://github.com/huggingface/lighteval) - LightEval 官方文档
- [Argilla 文档](https://docs.argilla.io) - 了解 Argilla 标注平台
- [MMLU 论文](https://arxiv.org/abs/2009.03300) - 关于 MMLU 测评基准的论文
- [LightEval 如何添加自定义任务](https://github.com/huggingface/lighteval/wiki/Adding-a-Custom-Task)
- [LightEval 如何添加自定义测评指标](https://github.com/huggingface/lighteval/wiki/Adding-a-New-Metric)
- [LightEval 如何使用现有的测评指标](https://github.com/huggingface/lighteval/wiki/Metric-List)