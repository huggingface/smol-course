# Quantization

This module will guide you through the concept of quantization which is useful for optimizing language models for efficient inference on CPUs, without the need for GPUs. We’ll focus on quantization for inference, a technique that reduces model size to improve inference speed. Additionally, we’ll explore how to perform inference on Intel and MLX (machine learning accelerators) CPUs, demonstrating how to leverage local resources for efficient and cost-effective model deployment.

## Quantization Fundamentals

First we will introduce quantization and explain how it reduces model size. Check out the [Fundamentals](./fundamentals.md) page for more information.

## The GGUF format

Second, we will introduce the GGUF format and the LlamaCPP package. We will explain how to quantize pre-trained or finetuned models, and how to use them for optimized inference with LlamaCPP. Check out the [GGUF](./gguf.md) page for more information.

## CPU Inference (Intel & MLX)

Finally, we will explore how to perform inference on Intel and MLX (machine learning accelerators for MacOS) CPUs, demonstrating how to leverage local resources for efficient and cost-effective model deployment. Check out the [CPU Inference](./cpu.md) page for more information.

## Exercise Notebooks

| Title | Description | Exercise | Link | Colab |
|-------|-------------|----------|------|-------|
| Quantization with LlamaCPP | Description| Exercise| [link](./notebooks/example.ipynb) | <a target="_blank" href="link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| CPU Inference (Intel or MLX) | Description| Exercise| [link](./notebooks/example.ipynb) | <a target="_blank" href="link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

## References

- [A Gentle Introduction to 8-bit Matrix Multiplication for transformers at scale using Hugging Face Transformers, Accelerate and bitsandbytes](https://huggingface.co/blog/hf-bitsandbytes-integration)
- [GGUF Docs](https://huggingface.co/docs/hub/gguf)
- [Mlx Docs](https://huggingface.co/docs/hub/mlx)
- [Intel IPEX](https://huggingface.co/docs/accelerate/usage_guides/ipex)
