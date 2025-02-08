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
| [Instruction Tuning](./1_instruction_tuning) | Learn supervised fine-tuning, chat templating, and basic instruction following | ✅ Ready | Dec 3, 2024 |
| [Preference Alignment](./2_preference_alignment) | Explore DPO and ORPO techniques for aligning models with human preferences | ✅ Ready  | Dec 6, 2024 |
| [Parameter-efficient Fine-tuning](./3_parameter_efficient_finetuning) | Learn LoRA, prompt tuning, and efficient adaptation methods | ✅ Ready | Dec 9, 2024 |
| [Evaluation](./4_evaluation) | Use automatic benchmarks and create custom domain evaluations | ✅ Ready | Dec 13, 2024 |
| [Vision-language Models](./5_vision_language_models) | Adapt multimodal models for vision-language tasks | ✅ Ready | Dec 16, 2024 |
| [Synthetic Datasets](./6_synthetic_datasets) | Create and validate synthetic datasets for training | ✅ Ready | Dec 20, 2024 |
| [Inference](./7_inference) | Infer with models efficiently | ✅ Ready | Jan 8, 2025 |
| [Agents](./8_agents) | Build your own agentic AI | ✅ Ready | Jan 13, 2025 ||

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
pip install transformers trl datasets huggingface_hub
```

