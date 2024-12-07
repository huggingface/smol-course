# Custom Domain Evaluation

When standard benchmarks don't cover your specific needs, you can design custom evaluation tasks. This guide will help you create evaluation pipelines tailored to your domain.

## Task Definition

1. **Evaluation Goals**
   - Define what aspects of the model you want to evaluate
   - Identify specific capabilities or behaviors to measure
   - Consider both positive and negative test cases

2. **Data Requirements**
   - Input format and structure
   - Expected output format
   - Edge cases and corner cases
   - Size of evaluation dataset needed

3. **Metrics Selection**
   - Choose metrics that align with your goals
   - Consider both automated and human evaluation metrics
   - Plan for statistical significance

## Implementation Guide

### Creating a Custom Task

```python
# Example of a custom LightEval task
from lighteval.tasks import Task

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

### Best Practices

1. **Documentation**
   - Document task objectives and methodology
   - Provide clear examples of inputs and outputs
   - Explain metric calculations and thresholds

2. **Validation**
   - Verify task correctness with small-scale tests
   - Include diverse test cases
   - Consider potential biases in your evaluation

3. **Maintenance**
   - Plan for dataset updates
   - Monitor for metric drift
   - Keep evaluation code maintainable

## Creating Evaluation Datasets

1. **Data Collection**
   - Gather domain-specific examples
   - Include edge cases and common scenarios
   - Consider data privacy and licensing

2. **Annotation**
   - Define clear annotation guidelines
   - Use tools like Argilla for efficient annotation
   - Ensure quality control measures

3. **Dataset Format**
   - Structure data for easy processing
   - Include metadata and documentation
   - Version control your datasets

# Next Steps
For a complete example of implementing a custom evaluation pipeline, see our [domain evaluation project](../project/README.md) which demonstrates these principles in practice.