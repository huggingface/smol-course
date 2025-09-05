# Synthetic Datasets

Synthetic data is artificially generated data that mimics real-world usage. It allows overcoming data limitations by expanding or enhancing datasets. Even though synthetic data was already used for some use cases, large language models have made synthetic datasets more popular for pre- and post-training, and the evaluation of language models.

We'll use [`distilabel`](https://distilabel.argilla.io/latest/), a framework for synthetic data and AI feedback for engineers who need fast, reliable and scalable pipelines based on verified research papers. For a deeper dive into the package and best practices, check out the [documentation](https://distilabel.argilla.io/latest/).

## Module Overview

Synthetic data for language models can be categorized into three taxonomies: instructions, preferences and critiques. We will focus on the first two categories, which focus on the generation of datasets for instruction tuning and preference alignment. In both categories, we will cover aspects of the third category, which focuses on improving existing data with model critiques and rewrites.

![Synthetic Data Taxonomies](./images/taxonomy-synthetic-data.png)

## Contents

### 1. [Instruction Datasets](./instruction_datasets.md)

Learn how to generate instruction datasets for instruction tuning. We will explore creating instruction tuning datasets thorugh basic prompting and using prompts more refined techniques from papers. Instruction tuning datasets with seed data for in-context learning can be created through methods like SelfInstruct and Magpie. Additionally, we will explore instruction evolution through EvolInstruct. [Start learning](./instruction_datasets.md).

### 2. [Preference Datasets](./preference_datasets.md)

Learn how to generate preference datasets for preference alignment. We will build on top of the methods and techniques introduced in section 1, by generating additional responses. Next, we will learn how to improve such responses with the EvolQuality prompt. Finally, we will explore how to evaluate responses with the the UltraFeedback prompt which will produce a score and critique, allowing us to create preference pairs. [Start learning](./preference_datasets.md).

### Exercise Notebooks

| Title | Description | Exercise | Link | Colab |
|-------|-------------|----------|------|-------|
| Instruction Dataset | Generate a dataset for instruction tuning | 🐢 Generate an instruction tuning dataset <br> 🐕 Generate a dataset for instruction tuning with seed data <br> 🦁 Generate a dataset for instruction tuning with seed data and with instruction evolution | [Link](./notebooks/instruction_sft_dataset.ipynb) | [Colab](https://githubtocolab.com/huggingface/smol-course/tree/main/6_synthetic_datasets/notebooks/instruction_sft_dataset.ipynb) |
| Preference Dataset | Generate a dataset for preference alignment | 🐢 Generate a preference alignment dataset <br> 🐕 Generate a preference alignment dataset with response evolution <br> 🦁 Generate a preference alignment dataset with response evolution and critiques  | [Link](./notebooks/preference_alignment_dataset.ipynb) | [Colab](https://githubtocolab.com/huggingface/smol-course/tree/main/6_synthetic_datasets/notebooks/preference_alignment_dataset.ipynb) |

## Resources

- [Distilabel Documentation](https://distilabel.argilla.io/latest/)
- [Synthetic Data Generator is UI app](https://huggingface.co/blog/synthetic-data-generator)
- [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)
- [Self-instruct](https://arxiv.org/abs/2212.10560)
- [Evol-Instruct](https://arxiv.org/abs/2304.12244)
- [Magpie](https://arxiv.org/abs/2406.08464)
- [UltraFeedback](https://arxiv.org/abs/2310.01377)
- [Deita](https://arxiv.org/abs/2312.15685)
