![smolcourse image](./banner.png)

# a smol course

This is a practical course on aligning language models for your specific use case. It's a handy way to get started with aligning language models, because everything runs on most local machines. There are minimal GPU requirements and no paid services. The course is based on the [SmolLM2](https://github.com/huggingface/smollm/tree/main) series of models, but you can transfer the skills you learn here to larger models or other small language models.

<a href="http://hf.co/join/discord">
<img src="https://img.shields.io/badge/Discord-7289DA?&logo=discord&logoColor=white"/>
</a>

<div style="background: linear-gradient(to right, #e0f7fa, #e1bee7, orange); padding: 20px; border-radius: 5px; margin-bottom: 20px; color: purple;">
    <h2>Participation is open, free, and now!</h2>
    <p>This course is open and peer reviewed. To get involved with the course <strong>open a pull request</strong> and submit your work for review. Here are the steps:</p>
    <ol>
        <li>Fork the repo <a href="https://github.com/huggingface/smol-course/fork">here</a></li>
        <li>Read the material, make changes, do the exercises, add your own examples.</li>
        <li>Open a PR on the december_2024 branch</li>
        <li>Get it reviewed and merged</li>
    </ol>
    <p>This should help you learn and to build a community-driven course that is always improving.</p>
</div>

We can discuss the process in this [discussion thread](https://github.com/huggingface/smol-course/discussions/2#discussion-7602932).

## Course Outline

This course provides a practical, hands-on approach to working with small language models, from initial training through to production deployment.

| Module | Description | Status | Release Date |
|--------|-------------|---------|--------------|
| [Instruction Tuning](./1_instruction_tuning) | Learn supervised fine-tuning, chat templating, and basic instruction following | ✅ Complete | Dec 3, 2024 |
| [Preference Alignment](./2_preference_alignment) | Explore DPO and ORPO techniques for aligning models with human preferences | ✅ Complete  | Dec 6, 2024 |
| [Parameter-efficient Fine-tuning](./3_parameter_efficient_finetuning) | Learn LoRA, prompt tuning, and efficient adaptation methods | [🚧 WIP](https://github.com/huggingface/smol-course/pull/41) | Dec 9, 2024 |
| [Evaluation](./4_evaluation) | Use automatic benchmarks and create custom domain evaluations | [🚧 WIP](https://github.com/huggingface/smol-course/issues/42) | Dec 13, 2024 |
| [Vision-language Models](./5_vision_language_models) | Adapt multimodal models for vision-language tasks | [🚧 WIP](https://github.com/huggingface/smol-course/issues/49) | Dec 16, 2024 |
| [Synthetic Datasets](./6_synthetic_datasets) | Create and validate synthetic datasets for training | 📝 Planned | Dec 20, 2024 |
| [Inference](./7_inference) | Infer with models efficiently | 📝 Planned | Dec 23, 2024 |

## Why Small Language Models?

While large language models have shown impressive capabilities, they often require significant computational resources and can be overkill for focused applications. Small language models offer several advantages for domain-specific applications:

- **Efficiency**: Require significantly less computational resources to train and deploy
- **Customization**: Easier to fine-tune and adapt to specific domains
- **Control**: Better understanding and control of model behavior
- **Cost**: Lower operational costs for training and inference
- **Privacy**: Can be run locally without sending data to external APIs
- **Green Technology**: Advocates efficient usage of resources with reduced carbon footprint
- **Easier Academic Research Development**: Provides an easy starter for academic research with cutting-edge LLMs with less logistical constraints

## Prerequisites

Before starting, ensure you have the following:
- Basic understanding of machine learning and natural language processing.
- Familiarity with Python, PyTorch, and the `transformers` library.
- Access to a pre-trained language model and a labeled dataset.

Here's my suggestion for the improved section, incorporating all your requirements and maintaining consistency:

## Quick Start with Dev Containers (Recommended)

The easiest way to get started is using a development container, avoiding any Python version or dependency conflicts. This gives you a ready-to-use environment with everything installed. Think of it as your "smol" setup—sandboxed and efficient!

### Prerequisites
- Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or alternative like [OrbStack](https://orbstack.dev/))

### Option 1: Visual Studio Code
1. Install the "Dev Containers" extension in VS Code
2. Fork this repository and clone it on your computer
3. When you open the repository in VS Code, click "Reopen in Container" in the bottom-right corner pop-up
4. VS Code will automatically set up the environment for you inside a container

### Option 2: Daytona (supports most code editors)
1. Install [Daytona](https://github.com/daytonaio/daytona/)
2. Run these commands:
```bash
daytona create https://github.com/huggingface/smol-course
daytona code smol-course
```
**Note:** With `daytona ide` command you can select your preferred editor like Cursor, Zed, JetBrains, or Jupyter.

Both options will provide you with an identical, isolated development environment that includes all the necessary tools and dependencies to work through this course.

**Note:** If you prefer to set up your environment manually, see the Installation section below.

## Installation

We maintain the course as a package so you can install dependencies easily via a package manager. We recommend [uv](https://github.com/astral-sh/uv) for this purpose, but you could use alternatives like `pip` or `pdm`.

### Using `uv`

With `uv` installed, you can install the course like this:

```bash
uv venv --python 3.11.0
uv sync
```

### Using `pip`

All the examples run in the same **python 3.11** environment, so you should create an environment and install dependencies like this:

```bash
# python -m venv .venv
# source .venv/bin/activate
pip install -r requirements.txt
```

### Google Colab

**From Google Colab** you will need to install dependencies flexibly based on the hardware you're using. Like this:

```bash
pip install -r transformers trl datasets huggingface_hub
```

## Engagement

Let's share this, so that loads of people can learn to finetune LLMs without expensive hardware.

[![Star History Chart](https://api.star-history.com/svg?repos=huggingface/smol-course&type=Date)](https://star-history.com/#huggingface/smol-course&Date)
