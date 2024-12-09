# Evaluation

This module covers evaluation approaches for your smol model, including both standard benchmarks and domain-specific evaluation methods. 

In this module we will use the library [`lighteval`](https://github.com/huggingface/lighteval). It's made at Hugging Face and it's integrated with the Hugging Face ecosystem. If you want to go deeper into the topic of evaluation with the authors of `lighteval`, you can check the evaluation [guidebook](https://github.com/huggingface/evaluation-guidebook).

## Module Overview

Evaluating language models focuses on assessing core capabilities:

- **Task Performance**: How well the model performs on specific tasks like question answering, summarization, etc.
- **Output Quality**: Measuring factors like coherence, relevance, and factual accuracy
- **Safety & Bias**: Checking for harmful outputs, biases, and toxic content
- **Domain Expertise**: Testing specialized knowledge and capabilities in specific fields

## Contents

### [Automatic Benchmarks](./automatic_benchmarks.md)
Learn how to evaluate your model using standardized benchmarks and metrics:
- Common benchmarks (MMLU, TruthfulQA, etc.)
- Evaluation metrics and settings
- Best practices for reproducible evaluation

### [Custom Domain Evaluation](./custom_evaluation.md)
Create custom evaluation pipelines for your specific use case:
- Designing evaluation tasks
- Implementing custom metrics
- Creating evaluation datasets

### [Domain Evaluation Project](./project/README.md)
A complete example of building a domain-specific evaluation pipeline:
- Generate evaluation datasets
- Annotate data with Argilla
- Create standardized datasets
- Evaluate models with LightEval

## Resources

- [Evaluation Guidebook](https://github.com/huggingface/evaluation-guidebook)
- [LightEval Documentation](https://github.com/huggingface/lighteval)
- [Argilla Documentation](https://docs.argilla.io)
