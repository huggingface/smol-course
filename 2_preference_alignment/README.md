# Preference Alignment

This module covers techniques for aligning language models with human preferences. While supervised fine-tuning helps models learn tasks, preference alignment encourages outputs to match human expectations and values.

## Overview

Typical alignment methods involve multiple stages:
1. Supervised Fine-Tuning (SFT) to adapt models to specific domains
2. Preference alignment (like RLHF or DPO) to improve response quality

Alternative approaches like ORPO combine instruction tuning and preference alignment into a single process. Here, we will focus on DPO and ORPO algorithms.

If you would like to learn more about the different alignment techniques, you can read more about them in the [Argilla Blog](https://argilla.io/blog/mantisnlp-rlhf-part-8). 

### 1️⃣ Direct Preference Optimization (DPO)

Direct Preference Optimization (DPO) simplifies preference alignment by directly optimizing models using preference data. This approach eliminates the need for separate reward models and complex reinforcement learning, making it more stable and efficient than traditional Reinforcement Learning from Human Feedback (RLHF). For more details, you can refer to the [Direct Preference Optimization (DPO) documentation](./dpo.md).


### 2️⃣ Odds Ratio Preference Optimization (ORPO)

ORPO introduces a combined approach to instruction tuning and preference alignment in a single process. It modifies the standard language modeling objective by combining negative log-likelihood loss with an odds ratio term on a token level. The approach features a unified single-stage training process, reference model-free architecture, and improved computational efficiency. ORPO has shown impressive results across various benchmarks, demonstrating better performance on AlpacaEval compared to traditional methods. For more details, you can refer to the [Odds Ratio Preference Optimization (ORPO) documentation](./orpo.md).

## Exercise Notebooks

| Title | Description | Exercise | Link | Colab |
|-------|-------------|----------|------|-------|
| DPO Training | Learn how to train models using Direct Preference Optimization | 🐢 Train a model using the Anthropic HH-RLHF dataset<br>🐕 Use your own preference dataset<br>🦁 Experiment with different preference datasets and model sizes | [Notebook](./notebooks/dpo_finetuning_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/2_preference_alignment/notebooks/dpo_finetuning_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| ORPO Training | Learn how to train models using Odds Ratio Preference Optimization | 🐢 Train a model using instruction and preference data<br>🐕 Experiment with different loss weightings<br>🦁 Compare ORPO results with DPO | [Notebook](./notebooks/orpo_finetuning_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/2_preference_alignment/notebooks/orpo_finetuning_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |


## Resources

- [TRL Documentation](https://huggingface.co/docs/trl/index) - Documentation for the Transformers Reinforcement Learning (TRL) library, which implements various alignment techniques including DPO.
- [DPO Paper](https://arxiv.org/abs/2305.18290) - Original research paper introducing Direct Preference Optimization as a simpler alternative to RLHF that directly optimizes language models using preference data.
- [ORPO Paper](https://arxiv.org/abs/2403.07691) - Introduces Odds Ratio Preference Optimization, a novel approach that combines instruction tuning and preference alignment in a single training stage.
- [Argilla RLHF Guide](https://argilla.io/blog/mantisnlp-rlhf-part-8/) - A guide explaining different alignment techniques including RLHF, DPO, and their practical implementations.
- [Blog post on DPO](https://huggingface.co/blog/dpo-trl) - Practical guide on implementing DPO using the TRL library with code examples and best practices.
- [TRL example script on DPO](https://github.com/huggingface/trl/blob/main/examples/scripts/dpo.py) - Complete example script demonstrating how to implement DPO training using the TRL library.
- [TRL example script on ORPO](https://github.com/huggingface/trl/blob/main/examples/scripts/orpo.py) - Reference implementation of ORPO training using the TRL library with detailed configuration options.
- [Hugging Face Alignment Handbook](https://github.com/huggingface/alignment-handbook) - Resource guides and codebase for aligning language models using various techniques including SFT, DPO, and RLHF.
