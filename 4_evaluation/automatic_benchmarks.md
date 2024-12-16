# Automatic Benchmarks

Automatic benchmarks serve as standardized tools for evaluating language models across different tasks and capabilities. While they provide a useful starting point for understanding model performance, it's important to recognize that they represent only one piece of a comprehensive evaluation strategy.

## Understanding Automatic Benchmarks

Automatic benchmarks typically consist of curated datasets with predefined tasks and evaluation metrics. These benchmarks aim to assess various aspects of model capability, from basic language understanding to complex reasoning. The key advantage of using automatic benchmarks is their standardization - they allow for consistent comparison across different models and provide reproducible results.

However, it's crucial to understand that benchmark performance doesn't always translate directly to real-world effectiveness. A model that excels at academic benchmarks may still struggle with specific domain applications or practical use cases.

## Benchmarks and Their Limitations

### General Knowledge Benchmarks

MMLU (Massive Multitask Language Understanding) tests knowledge across 57 subjects, from science to humanities. While comprehensive, it may not reflect the depth of expertise needed for specific domains. TruthfulQA evaluates a model's tendency to reproduce common misconceptions, though it can't capture all forms of misinformation.

### Reasoning Benchmarks
BBH (Big Bench Hard) and GSM8K focus on complex reasoning tasks. BBH tests logical thinking and planning, while GSM8K specifically targets mathematical problem-solving. These benchmarks help assess analytical capabilities but may not capture the nuanced reasoning required in real-world scenarios.

### Language Understanding
HELM provides a holistic evaluation framework, while WinoGrande tests common sense through pronoun disambiguation. These benchmarks offer insights into language processing capabilities but may not fully represent the complexity of natural conversation or domain-specific terminology.

## Alternative Evaluation Approaches

Many organizations have developed alternative evaluation methods to address the limitations of standard benchmarks:

### LLM-as-Judge
Using one language model to evaluate another's outputs has become increasingly popular. This approach can provide more nuanced feedback than traditional metrics, though it comes with its own biases and limitations.

### Evaluation Arenas
Platforms like Anthropic's Constitutional AI Arena allow models to interact and evaluate each other in controlled environments. This can reveal strengths and weaknesses that might not be apparent in traditional benchmarks.

### Custom Benchmark Suites
Organizations often develop internal benchmark suites tailored to their specific needs and use cases. These might include domain-specific knowledge tests or evaluation scenarios that mirror actual deployment conditions.

## Creating Your Own Evaluation Strategy

Remember that while LightEval makes it easy to run standard benchmarks, you should also invest time in developing evaluation methods specific to your use case.

While standard benchmarks provide a useful baseline, they shouldn't be your only evaluation method. Here's how to develop a more comprehensive approach:

1. Start with relevant standard benchmarks to establish a baseline and enable comparison with other models.

2. Identify the specific requirements and challenges of your use case. What tasks will your model actually perform? What kinds of errors would be most problematic?

3. Develop custom evaluation datasets that reflect your actual use case. This might include:
   - Real user queries from your domain
   - Common edge cases you've encountered
   - Examples of particularly challenging scenarios

4. Consider implementing a multi-layered evaluation strategy:
   - Automated metrics for quick feedback
   - Human evaluation for nuanced understanding
   - Domain expert review for specialized applications
   - A/B testing in controlled environments

## Using LightEval for Benchmarking

LightEval tasks are defined using a specific format:
```
{suite}|{task}|{num_few_shot}|{auto_reduce}
```

- **suite**: The benchmark suite (e.g., 'mmlu', 'truthfulqa')
- **task**: Specific task within the suite (e.g., 'abstract_algebra')
- **num_few_shot**: Number of examples to include in prompt (0 for zero-shot)
- **auto_reduce**: Whether to automatically reduce few-shot examples if prompt is too long (0 or 1)

Example: `"mmlu|abstract_algebra|0|0"` evaluates on MMLU's abstract algebra task with zero-shot inference.

### Example Evaluation Pipeline

Here's a complete example of setting up and running an evaluation on automatic benchmarks relevant to one specific domain:

```python
from lighteval.tasks import Task, Pipeline
from transformers import AutoModelForCausalLM

# Define tasks to evaluate
domain_tasks = [
    "mmlu|anatomy|0|0",
    "mmlu|high_school_biology|0|0", 
    "mmlu|high_school_chemistry|0|0",
    "mmlu|professional_medicine|0|0"
]

# Configure pipeline parameters
pipeline_params = {
    "max_samples": 40,  # Number of samples to evaluate
    "batch_size": 1,    # Batch size for inference
    "num_workers": 4    # Number of worker processes
}

# Create evaluation tracker
evaluation_tracker = EvaluationTracker(
    output_path="./results",
    save_generations=True
)

# Load model and create pipeline
model = AutoModelForCausalLM.from_pretrained("your-model-name")
pipeline = Pipeline(
    tasks=domain_tasks,
    pipeline_parameters=pipeline_params,
    evaluation_tracker=evaluation_tracker,
    model=model
)

# Run evaluation
pipeline.evaluate()

# Get and display results
results = pipeline.get_results()
pipeline.show_results()
```

Results are displayed in a tabular format showing:
```
|                  Task                  |Version|Metric|Value |   |Stderr|
|----------------------------------------|------:|------|-----:|---|-----:|
|all                                     |       |acc   |0.3333|±  |0.1169|
|leaderboard:mmlu:_average:5             |       |acc   |0.3400|±  |0.1121|
|leaderboard:mmlu:anatomy:5              |      0|acc   |0.4500|±  |0.1141|
|leaderboard:mmlu:high_school_biology:5  |      0|acc   |0.1500|±  |0.0819|
```

You can also handle the results in a pandas DataFrame and visualise or represent them as you want.

# Next Steps

⏩ Explore [Custom Domain Evaluation](./custom_evaluation.md) to learn how to create evaluation pipelines tailored to your specific needs
