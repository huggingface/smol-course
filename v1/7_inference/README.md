# Inference

Inference is the process of using a trained language model to generate predictions or responses. While inference might seem straightforward, deploying models efficiently at scale requires careful consideration of various factors like performance, cost, and reliability. Large Language Models (LLMs) present unique challenges due to their size and computational requirements.

We'll explore both simple and production-ready approaches using the [`transformers`](https://huggingface.co/docs/transformers/index) library and [`text-generation-inference`](https://github.com/huggingface/text-generation-inference), two popular frameworks for LLM inference. For production deployments, we'll focus on Text Generation Inference (TGI), which provides optimized serving capabilities.

## Module Overview

LLM inference can be categorized into two main approaches: simple pipeline-based inference for development and testing, and optimized serving solutions for production deployments. We'll cover both approaches, starting with the simpler pipeline approach and moving to production-ready solutions.

## Contents

### 1. [Basic Pipeline Inference](./pipeline_inference.md)

Learn how to use the Hugging Face Transformers pipeline for basic inference. We'll cover setting up pipelines, configuring generation parameters, and best practices for local development. The pipeline approach is perfect for prototyping and small-scale applications. [Start learning](./pipeline_inference.md).

### 2. [Production Inference with TGI](./tgi_inference.md)

Learn how to deploy models for production using Text Generation Inference. We'll explore optimized serving techniques, batching strategies, and monitoring solutions. TGI provides production-ready features like health checks, metrics, and Docker deployment options. [Start learning](./text_generation_inference.md).

### Exercise Notebooks

| Title | Description | Exercise | Link | Colab |
|-------|-------------|----------|------|-------|
| Pipeline Inference | Basic inference with transformers pipeline | 🐢 Set up a basic pipeline <br> 🐕 Configure generation parameters <br> 🦁 Create a simple web server | [Link](./notebooks/basic_pipeline_inference.ipynb) | [Colab](https://githubtocolab.com/huggingface/smol-course/tree/main/7_inference/notebooks/basic_pipeline_inference.ipynb) |
| TGI Deployment | Production deployment with TGI | 🐢 Deploy a model with TGI <br> 🐕 Configure performance optimizations <br> 🦁 Set up monitoring and scaling | [Link](./notebooks/tgi_deployment.ipynb) | [Colab](https://githubtocolab.com/huggingface/smol-course/tree/main/7_inference/notebooks/tgi_deployment.ipynb) |

## Resources

- [Hugging Face Pipeline Tutorial](https://huggingface.co/docs/transformers/en/pipeline_tutorial)
- [Text Generation Inference Documentation](https://huggingface.co/docs/text-generation-inference/en/index)
- [Pipeline WebServer Guide](https://huggingface.co/docs/transformers/en/pipeline_tutorial#using-pipelines-for-a-webserver)
- [TGI GitHub Repository](https://github.com/huggingface/text-generation-inference)
- [Hugging Face Model Deployment Documentation](https://huggingface.co/docs/inference-endpoints/index)
- [vLLM: High-throughput LLM Serving](https://github.com/vllm-project/vllm)
- [Optimizing Transformer Inference](https://huggingface.co/blog/optimize-transformer-inference)
