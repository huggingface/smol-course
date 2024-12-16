# Custom Domain Evaluation

While standard benchmarks provide valuable insights, many applications require specialized evaluation approaches tailored to specific domains or use cases. This guide will help you create custom evaluation pipelines that accurately assess your model's performance in your target domain.

## Designing Your Evaluation Strategy

A successful custom evaluation strategy starts with clear objectives. Consider what specific capabilities your model needs to demonstrate in your domain. This might include technical knowledge, reasoning patterns, or domain-specific formats. Document these requirements carefully - they'll guide both your task design and metric selection.

Your evaluation should test both standard use cases and edge cases. For example, in a medical domain, you might evaluate both common diagnostic scenarios and rare conditions. In financial applications, you might test both routine transactions and complex edge cases involving multiple currencies or special conditions.

## Implementation with LightEval

LightEval provides a flexible framework for implementing custom evaluations. Here's how to create a custom task:

```python
from lighteval.tasks import Task, Doc
from lighteval.metrics import SampleLevelMetric, MetricCategory, MetricUseCase

class CustomEvalTask(Task):
    def __init__(self):
        super().__init__(
            name="custom_task",
            version="0.0.1",
            metrics=["accuracy", "f1"],  # Your chosen metrics
            description="Description of your custom evaluation task"
        )
    
    def get_prompt(self, sample):
        # Format your input into a prompt
        return f"Question: {sample['question']}\nAnswer:"
    
    def process_response(self, response, ref):
        # Process model output and compare to reference
        return response.strip() == ref.strip()
```

## Custom Metrics

Domain-specific tasks often require specialized metrics. LightEval provides a flexible framework for creating custom metrics that capture domain-relevant aspects of performance:

```python
from aenum import extend_enum
from lighteval.metrics import Metrics, SampleLevelMetric, SampleLevelMetricGrouping
import numpy as np

# Define a sample-level metric function
def custom_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> dict:
    """Example metric that returns multiple scores per sample"""
    response = predictions[0]
    return {
        "accuracy": response == formatted_doc.choices[formatted_doc.gold_index],
        "length_match": len(response) == len(formatted_doc.reference)
    }

# Create a metric that returns multiple values per sample
custom_metric_group = SampleLevelMetricGrouping(
    metric_name=["accuracy", "length_match"],  # Names of sub-metrics
    higher_is_better={  # Whether higher values are better for each metric
        "accuracy": True,
        "length_match": True
    },
    category=MetricCategory.CUSTOM,
    use_case=MetricUseCase.SCORING,
    sample_level_fn=custom_metric,
    corpus_level_fn={  # How to aggregate each metric
        "accuracy": np.mean,
        "length_match": np.mean
    }
)

# Register the metric with LightEval
extend_enum(Metrics, "custom_metric_name", custom_metric_group)
```

For simpler cases where you only need one metric value per sample:

```python
def simple_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> bool:
    """Example metric that returns a single score per sample"""
    response = predictions[0]
    return response == formatted_doc.choices[formatted_doc.gold_index]

simple_metric_obj = SampleLevelMetric(
    metric_name="simple_accuracy",
    higher_is_better=True,
    category=MetricCategory.CUSTOM,
    use_case=MetricUseCase.SCORING,
    sample_level_fn=simple_metric,
    corpus_level_fn=np.mean  # How to aggregate across samples
)

extend_enum(Metrics, "simple_metric", simple_metric_obj)
```

You can then use your custom metrics in your evaluation tasks by referencing them in the task configuration. The metrics will be automatically computed across all samples and aggregated according to your specified functions.

For more complex metrics, consider:
- Using metadata in your formatted documents to weight or adjust scores
- Implementing custom aggregation functions for corpus-level statistics
- Adding validation checks for your metric inputs
- Documenting edge cases and expected behavior

For a complete example of custom metrics in action, see our [domain evaluation project](./project/README.md).

## Dataset Creation

High-quality evaluation requires carefully curated datasets. Consider these approaches for dataset creation:

1. Expert Annotation: Work with domain experts to create and validate evaluation examples. Tools like [Argilla](https://github.com/argilla-io/argilla) make this process more efficient.

2. Real-World Data: Collect and anonymize real usage data, ensuring it represents actual deployment scenarios.

3. Synthetic Generation: Use LLMs to generate initial examples, then have experts validate and refine them.

## Best Practices

- Document your evaluation methodology thoroughly, including any assumptions or limitations
- Include diverse test cases that cover different aspects of your domain
- Consider both automated metrics and human evaluation where appropriate
- Version control your evaluation datasets and code
- Regularly update your evaluation suite as you discover new edge cases or requirements

## References

- [LightEval Custom Task Guide](https://github.com/huggingface/lighteval/wiki/Adding-a-Custom-Task)
- [LightEval Custom Metrics](https://github.com/huggingface/lighteval/wiki/Adding-a-New-Metric)
- [Argilla Documentation](https://docs.argilla.io) for dataset annotation
- [Evaluation Guidebook](https://github.com/huggingface/evaluation-guidebook) for general evaluation principles

# Next Steps

⏩ For a complete example of implementing these concepts, see our [domain evaluation project](./project/README.md).