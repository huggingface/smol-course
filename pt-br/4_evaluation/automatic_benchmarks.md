# Automatic Benchmarks

Automatic benchmarks provide a standardized way to evaluate language models across different tasks and capabilities. These benchmarks use predefined datasets and metrics to assess model performance.

The advantage of automatic benchmarks is that they are standardized and can be used to compare models across different domains. The disadvantage is that they may not cover all the capabilities of your use case.

## Key Components

1. **Tasks**: Predefined problems the model needs to solve
   - **Multiple Choice**
     - Question answering with predefined options
     - Subject knowledge testing (e.g., MMLU)
     - Common sense reasoning (e.g., WinoGrande)
   
   - **Text Generation**
     - Open-ended question answering
     - Summarization
     - Translation
     - Story completion
   
   - **Classification**
     - Sentiment analysis
     - Topic classification
     - Natural language inference
     - Toxicity detection
   
   - **Question Answering**
     - Extractive QA (finding answers in context)
     - Generative QA (formulating new answers)
     - Multi-hop reasoning
     - Math word problems

2. **Metrics**: Quantitative measures of performance
   - **Accuracy-based**
     - Exact match accuracy
     - Multiple choice accuracy
     - F1 score for partial matches
     - Matthews correlation coefficient
   
   - **Generation Quality**
     - ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
     - BLEU (Bilingual Evaluation Understudy)
     - METEOR (for machine translation)
     - BERTScore for semantic similarity
   
   - **Specialized Metrics**
     - Perplexity for language modeling
     - Code execution accuracy
     - Factual consistency scores
     - Toxicity measures

3. **Evaluation Settings**
   - Zero-shot: No examples provided
   - Few-shot: Limited examples in prompt
   - Fine-tuned: Model adapted to task
   - Chain-of-thought: Reasoning steps included

## Common Benchmarks

LightEval supports various established benchmarks that test the main capabilities of LLMs. Here is a non-exhaustive list:

### General Knowledge
- **MMLU** (Massive Multitask Language Understanding): Tests knowledge across 57 subjects including science, humanities, math, and more
- **TruthfulQA**: Evaluates model's ability to avoid common misconceptions and false statements
- **ARC**: Grade-school level science questions requiring reasoning

### Reasoning
- **BBH** (Big Bench Hard): Tests complex reasoning including logic, planning, and common sense
- **GSM8K**: Grade school math word problems requiring multi-step reasoning

### Language Understanding
- **HELM**: Holistic evaluation framework covering multiple capabilities
- **WinoGrande**: Tests common sense reasoning through pronoun disambiguation

### Code and Math
- **HumanEval**: Programming problems testing code generation capabilities
- **MATH**: High school and competition level mathematics problems

For a comprehensive list of evaluation datasets and their detailed descriptions, see the [Evaluation Datasets Catalog](https://github.com/huggingface/evaluation-guidebook/blob/main/contents/automated-benchmarks/some-evaluation-datasets.md) in the Evaluation Guidebook.

## Using LightEval

### Task Structure

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

Here's a complete example of setting up and running an evaluation:

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

### Show Results

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

- [Custom Evaluation](./custom_evaluation.md)
- [Domain Evaluation Project](./project/README.md)