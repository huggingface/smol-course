# Quantization

This module will guide you through optimizing language models for efficient inference on CPUs, without the need for heavy GPUs.
We’ll cover quantization, a technique that reduces model size and improves inference speed, and introduce GGUF (a format for optimized models).
Additionally, we’ll explore how to perform inference on Intel and MLX (machine learning accelerators) CPUs, demonstrating how to leverage local resources for efficient and cost-effective model deployment.

## Quantization

TBD
Motivation? less memory less accuracy? comparing the results? Int4, Int8, bf16?

## GGUF format

TBD
using huggingface to run diff quantization models
ollama and llm.cpp?

## CPU Inference (Intel & MLX)

TBD
use mlx for inference
use intel for inference (ipex? openvino?)

## Exercise Notebooks

| Title | Description | Exercise | Link | Colab |
|-------|-------------|----------|------|-------|
| Quantization | Description| Exercise| [link](./notebooks/example.ipynb) | <a target="_blank" href="link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| GGUF format | Description| Exercise| [link](./notebooks/example.ipynb) | <a target="_blank" href="link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| CPU Inference (Intel & MLX) | Description| Exercise| [link](./notebooks/example.ipynb) | <a target="_blank" href="link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

## References

- [A Gentle Introduction to 8-bit Matrix Multiplication for transformers at scale using Hugging Face Transformers, Accelerate and bitsandbytes](https://huggingface.co/blog/hf-bitsandbytes-integration)
- [GGUF Docs](https://huggingface.co/docs/hub/gguf)
- [Mlx Docs](https://huggingface.co/docs/hub/mlx)
- [Intel IPEX](https://huggingface.co/docs/accelerate/usage_guides/ipex)
