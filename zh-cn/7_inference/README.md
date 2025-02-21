# 大模型推理

推理是指用经过训练的语言模型来进行预测或生成回答的过程。尽管推理看似简单直接，但要大规模高效部署模型，就需要仔细考量诸如性能、成本和可靠性等多种因素。由于大语言模型（LLMs）的规模和计算需求，模型推理存在着一些独特的挑战。

这里，我们借助 [`transformers`](https://huggingface.co/docs/transformers/index) 和 [`text-generation-inference`](https://github.com/huggingface/text-generation-inference) 来探索简单的推理方法以及适用于生产环境的推理方法。针对生产环境的模型部署，我们将重点关注 Text Generation Inference 这个工具包（简称 TGI），它提供了优化过的服务能力。

## 章节概览

大语言模型的推理可以分为两类：简单的使用 pipeline 进行推理，这适用于开发和测试阶段；优化过的服务级推理方案，者适用于生产环境的部署。我们将会讲解这两类方法，从简单的 pipeline 方法开始，再逐步深入到生产环境部署方案。

## 内容目录

### 1. [基本的 Pipeline 推理](./inference_pipeline_cn.md)

这一节你将了解使用 Hugging Face Transformers 的 pipeline 进行基本的推理方法。我们将讲解 pipeline 的设置、生成参数的配置，以及本地部署的最佳实践。该方法适用于原型设计和小规模应用。

### 2. [使用 TGI 进行生产环境部署](./text_generation_inference_cn.md)

这一节你讲学习如何使用 Text Generation Inference 这个工具包进行生产环境的模型部署。我们将探索服务端模型部署的优化技术、组 batch 推理的策略，以及如何监控。TGI 提供了很多适用于生产环境的功能，如健康监测、评估指标、Docker 部署等。

### 练习

| 标题 | 概述 | 练习 | 链接 | Colab |
|-------|-------------|----------|------|-------|
| 使用 Pipeline 推理 | 使用 transformers 的 pipeline 进行推理 | 🐢 创建一个 pipeline <br> 🐕 配置生成参数 <br> 🦁 创建一个简单的网页端服务 | [Link](./notebooks/basic_pipeline_inference_cn.ipynb) | [Colab](https://githubtocolab.com/huggingface/smol-course/tree/main/7_inference/notebooks/basic_pipeline_inference.ipynb) |
| 使用 TGI 部署 | 在生产环境进行 TGI 部署 | 🐢 用 TGI 部署一个模型 <br> 🐕 调节参数，优化性能 <br> 🦁 监控和扩展 | [Link](./notebooks/tgi_deployment_cn.ipynb) | [Colab](https://githubtocolab.com/huggingface/smol-course/tree/main/7_inference/notebooks/tgi_deployment.ipynb) |

## 参考资料

- [Hugging Face 的 Pipeline 教程](https://huggingface.co/docs/transformers/en/pipeline_tutorial)
- [Text Generation Inference 官方文档](https://huggingface.co/docs/text-generation-inference/en/index)
- [基于 Pipeline 创建 WebServer 指南](https://huggingface.co/docs/transformers/en/pipeline_tutorial#using-pipelines-for-a-webserver)
- [TGI GitHub 代码仓库](https://github.com/huggingface/text-generation-inference)
- [Hugging Face 模型部署文档](https://huggingface.co/docs/inference-endpoints/index)
- [vLLM: High-throughput LLM Serving](https://github.com/vllm-project/vllm)
- [优化 Transformer 模型推理](https://huggingface.co/blog/optimize-transformer-inference)
