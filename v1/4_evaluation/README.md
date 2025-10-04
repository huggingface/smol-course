# Evaluation

Evaluation is a critical step in developing and deploying language models. It helps us understand how well our models perform across different capabilities and identify areas for improvement. This module covers both standard benchmarks and domain-specific evaluation approaches to comprehensively assess your smol model.

We'll use [`lighteval`](https://github.com/huggingface/lighteval), a powerful evaluation library developed by Hugging Face that integrates seamlessly with the Hugging Face ecosystem. For a deeper dive into evaluation concepts and best practices, check out the evaluation [guidebook](https://github.com/huggingface/evaluation-guidebook).

## Module Overview 

A thorough evaluation strategy examines multiple aspects of model performance. We assess task-specific capabilities like question answering and summarization to understand how well the model handles different types of problems. We measure output quality through factors like coherence and factual accuracy. Safety evaluation helps identify potential harmful outputs or biases. Finally, domain expertise testing verifies the model's specialized knowledge in your target field.

## Contents

### 1️⃣ [Automatic Benchmarks](./automatic_benchmarks.md)

Learn to evaluate your model using standardized benchmarks and metrics. We'll explore common benchmarks like MMLU and TruthfulQA, understand key evaluation metrics and settings, and cover best practices for reproducible evaluation.


### 2️⃣ [Custom Domain Evaluation](./custom_evaluation.md)
Discover how to create evaluation pipelines tailored to your specific use case. We'll walk through designing custom evaluation tasks, implementing specialized metrics, and building evaluation datasets that match your requirements.

### 3️⃣ [Domain Evaluation Project](./project/README.md)
Follow a complete example of building a domain-specific evaluation pipeline. You'll learn to generate evaluation datasets, use Argilla for data annotation, create standardized datasets, and evaluate models using LightEval.

### Exercise Notebooks

| Title | Description | Exercise | Link | Colab |
|-------|-------------|----------|------|-------|
| Evaluate and Analyze Your LLM | Learn how to use LightEval to evaluate and compare models on specific domains | 🐢 Use medical domain tasks to evaluate a model <br> 🐕 Create a new domain evaluation with different MMLU tasks <br> 🦁 Create a custom evaluation task for your domain | [Notebook](./notebooks/lighteval_evaluate_and_analyse_your_LLM.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/4_evaluation/notebooks/lighteval_evaluate_and_analyse_your_LLM.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |


## Resources

- [Evaluation Guidebook](https://github.com/huggingface/evaluation-guidebook) - Comprehensive guide to LLM evaluation
- [LightEval Documentation](https://github.com/huggingface/lighteval) - Official docs for the LightEval library
- [Argilla Documentation](https://docs.argilla.io) - Learn about the Argilla annotation platform
- [MMLU Paper](https://arxiv.org/abs/2009.03300) - Paper describing the MMLU benchmark
- [Creating a Custom Task](https://github.com/huggingface/lighteval/wiki/Adding-a-Custom-Task)
- [Creating a Custom Metric](https://github.com/huggingface/lighteval/wiki/Adding-a-New-Metric)
- [Using existing metrics](https://github.com/huggingface/lighteval/wiki/Metric-List)