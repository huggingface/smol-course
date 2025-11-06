# Introduction to Instruction Tuning with SmolLM3

Welcome to the smollest course of fine-tuning! This module will guide you through instruction tuning using **SmolLM3**, Hugging Face's latest 3B parameter model that achieves state-of-the-art performance for its size, while remaining accessible for learning and experimentation.

> [!TIP]
> By the end of this course you will be fine tuning an LLM with SFT. This course is smol but fast! If you're like for a smoother gradient, check out the [The LLM Course](https://huggingface.co/learn/llm-course/chapter1/1).
>
> After completing this unit (and the assignment), don’t forget to test your knowledge with the [quiz](https://huggingface.co/spaces/smol-course/unit_1_quiz)!

## What is Instruction Tuning?

Instruction tuning is the process of adapting pre-trained language models to follow human instructions and engage in conversations. While base models like [`SmolLM3-3B-Base`](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base) are trained to predict the next token, instruction-tuned models like [`SmolLM3-3B`](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) are specifically trained to:

- Follow user instructions accurately
- Engage in natural conversations
- Provide helpful, harmless, and honest responses
- Maintain context across multi-turn interactions
- Use tools or MCP servers to perform tasks

This transformation from a text completion model to an instruction-following assistant is achieved through supervised fine-tuning on carefully curated datasets.

> [!TIP]
> We dive into the instruction tuning [here](https://huggingface.co/learn/llm-course/en/chapter11/2) in the LLM Course.

## Why SmolLM3 for Learning?

[SmolLM3](https://hf.co/blog/smollm3) is perfect for learning instruction tuning because it:

- Fits in a single GPU at a reasonable cost 
- Achieves competitive performance
- Supports multilingual conversations
- Supports extended context length up to 8k tokens (with some variants supporting longer contexts up to 128k tokens)
- Features dual-mode reasoning with explicit thinking capabilities
- Comes with complete training recipes so you understand exactly how it was built

## Module Overview

In this comprehensive module, we will explore four key areas:

### Chat Templates

Chat templates are the foundation of instruction tuning - they structure interactions between users and AI models, ensuring consistent and contextually appropriate responses. You'll learn:

- How SmolLM3's chat template works
- Converting conversations to the proper format
- Working with system prompts and multi-turn conversations
- Using the Transformers library's built-in template support

For detailed information, see [Chat Templates](./2).

### Supervised Fine-Tuning (SFT)

Supervised Fine-Tuning is the core technique for adapting pre-trained models to follow instructions. You'll master:

- The theory behind SFT and when to use it
- Working with the SmolTalk2 dataset
- Using TRL's [`SFTTrainer`](https://huggingface.co/docs/trl/en/sft_trainer) for efficient training
- Best practices for data preparation and training configuration

For a comprehensive guide, see [Supervised Fine-Tuning](./3).

### Hands-on Exercises

Put your knowledge into practice with progressively challenging exercises:

- Process datasets for instruction tuning
- Fine-tune SmolLM3 on different tasks
- Use both Python APIs and CLI tools
- Compare base model vs fine-tuned model performance

Complete exercises and examples are in [Exercises](./4).

### Hugging Face Jobs

Hugging Face [Jobs](https://huggingface.co/docs/huggingface_hub/en/guides/jobs) is a fully managed cloud infrastructure for training models without the hassle of setting up GPUs, managing dependencies, or configuring environments locally. This is particularly valuable for SFT training, which can be resource-intensive and time-consuming.

For a comprehensive guide, see [Hugging Face Jobs](./5).

## What You'll Build

By the end of this module, you'll have:

- Fine-tuned your own SmolLM3 model on a custom dataset
- Understanding of chat template formatting and conversation structure
- Experience with both programmatic and CLI-based training workflows
- A model deployed to Hugging Face Hub that others can use
- Foundation knowledge for more advanced fine-tuning techniques

Let's dive into the fascinating world of instruction tuning!

## References

- [Transformers documentation on chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Script for Supervised Fine-Tuning in TRL](https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py)
- [`SFTTrainer` in TRL](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [Direct Preference Optimization Paper](https://huggingface.co/papers/2305.18290)
- [How to fine-tune Google Gemma with ChatML and Hugging Face TRL](https://www.philschmid.de/fine-tune-google-gemma)
- [Fine-tuning LLM to Generate Persian Product Catalogs in JSON Format](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_to_generate_persian_product_catalogs_in_json_format) 