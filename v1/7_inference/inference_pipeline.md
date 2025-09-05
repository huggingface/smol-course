# Basic Inference with Transformers Pipeline

The `pipeline` abstraction in 🤗 Transformers provides a simple way to run inference with any model from the Hugging Face Hub. It handles all the preprocessing and postprocessing steps, making it easy to use models without deep knowledge of their architecture or requirements.

## How Pipelines Work

Hugging Face pipelines streamline the machine learning workflow by automating three critical stages between raw input and human-readable output:

**Preprocessing Stage**
The pipeline first prepares your raw inputs for the model. This varies by input type:
- Text inputs undergo tokenization to convert words into model-friendly token IDs
- Images are resized and normalized to match model requirements
- Audio is processed through feature extraction to create spectrograms or other representations

**Model Inference**
During the forward pass, the pipeline:
- Handles batching of inputs automatically for efficient processing
- Places computation on the optimal device (CPU/GPU)
- Applies performance optimizations like half-precision (FP16) inference where supported

**Postprocessing Stage**
Finally, the pipeline converts raw model outputs into useful results:
- Decodes token IDs back into readable text
- Transforms logits into probability scores
- Formats outputs according to the specific task (e.g., classification labels, generated text)

This abstraction lets you focus on your application logic while the pipeline handles the technical complexity of model inference.

## Basic Usage

Here's how to use a pipeline for text generation:

```python
from transformers import pipeline

# Create a pipeline with a specific model
generator = pipeline(
    "text-generation",
    model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

# Generate text
response = generator(
    "Write a short poem about coding:",
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7
)
print(response[0]['generated_text'])
```

## Key Configuration Options

### Model Loading
```python
# CPU inference
generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-1.7B-Instruct", device="cpu")

# GPU inference (device 0)
generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-1.7B-Instruct", device=0)

# Automatic device placement
generator = pipeline(
    "text-generation",
    model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    device_map="auto",
    torch_dtype="auto"
)
```

### Generation Parameters

```python
response = generator(
    "Translate this to French:",
    max_new_tokens=100,     # Maximum length of generated text
    do_sample=True,         # Use sampling instead of greedy decoding
    temperature=0.7,        # Control randomness (higher = more random)
    top_k=50,               # Limit to top k tokens
    top_p=0.95,             # Nucleus sampling threshold
    num_return_sequences=1  # Number of different generations
)
```

## Processing Multiple Inputs

Pipelines can efficiently handle multiple inputs through batching:

```python
# Prepare multiple prompts
prompts = [
    "Write a haiku about programming:",
    "Explain what an API is:",
    "Write a short story about a robot:"
]

# Process all prompts efficiently
responses = generator(
    prompts,
    batch_size=4,              # Number of prompts to process together
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7
)

# Print results
for prompt, response in zip(prompts, responses):
    print(f"Prompt: {prompt}")
    print(f"Response: {response[0]['generated_text']}\n")
```

## Web Server Integration

Here's how to integrate a pipeline into a FastAPI application:

```python
from fastapi import FastAPI, HTTPException
from transformers import pipeline
import uvicorn

app = FastAPI()

# Initialize pipeline globally
generator = pipeline(
    "text-generation",
    model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    device_map="auto"
)

@app.post("/generate")
async def generate_text(prompt: str):
    try:
        if not prompt:
            raise HTTPException(status_code=400, detail="No prompt provided")
            
        response = generator(
            prompt,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7
        )
        
        return {"generated_text": response[0]['generated_text']}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
```

## Limitations

While pipelines are great for prototyping and small-scale deployments, they have some limitations:

- Limited optimization options compared to dedicated serving solutions
- No built-in support for advanced features like dynamic batching
- May not be suitable for high-throughput production workloads

For production deployments with high throughput requirements, consider using Text Generation Inference (TGI) or other specialized serving solutions.

## Resources

- [Hugging Face Pipeline Tutorial](https://huggingface.co/docs/transformers/en/pipeline_tutorial)
- [Pipeline API Reference](https://huggingface.co/docs/transformers/en/main_classes/pipelines)
- [Text Generation Parameters](https://huggingface.co/docs/transformers/en/main_classes/text_generation)
- [Model Quantization Guide](https://huggingface.co/docs/transformers/en/perf_infer_gpu_one)