# Text Generation Inference (TGI)

Text Generation Inference (TGI) is a toolkit developed by Hugging Face for deploying and serving Large Language Models (LLMs). It's designed to enable high-performance text generation for popular open-source LLMs. TGI is used in production by Hugging Chat - An open-source interface for open-access models.

## Why Use Text Generation Inference?

Text Generation Inference addresses the key challenges of deploying large language models in production. While many frameworks excel at model development, TGI specifically optimizes for production deployment and scaling. Some key features include:

- **Tensor Parallelism**: TGI's can split models across multiple GPUs through tensor parallelism, essential for serving larger models efficiently.
- **Continuous Batching**: The continuous batching system maximizes GPU utilization by dynamically processing requests, while optimizations like Flash Attention and Paged Attention significantly reduce memory usage and increase speed.
- **Token Streaming**: Real-time applications benefit from token streaming via Server-Sent Events, delivering responses with minimal latency.

## How to Use Text Generation Inference

### Basic Python Usage

TGI uses a simple yet powerful REST API integration which makes it easy to integrate with your applications. 

### Using the REST API

TGI exposes a RESTful API that accepts JSON payloads. This makes it accessible from any programming language or tool that can make HTTP requests. Here's a basic example using curl:

```bash
# Basic generation request
curl localhost:8080/v1/chat/completions \
    -X POST \
    -d '{
  "model": "tgi",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What is deep learning?"
    }
  ],
  "stream": true,
  "max_tokens": 20
}' \
    -H 'Content-Type: application/json'
```

### Using the `huggingface_hub` Python Client

The `huggingface_hub` python client client handles connection management, request formatting, and response parsing. Here's how to get started.

```python
from huggingface_hub import InferenceClient

client = InferenceClient(
    base_url="http://localhost:8080/v1/",
)

output = client.chat.completions.create(
    model="tgi",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Count to 10"},
    ],
    stream=True,
    max_tokens=1024,
)

for chunk in output:
    print(chunk.choices[0].delta.content)
```


### Using OpenAI API

Many libraries support the OpenAI API, so you can use the same client to interact with TGI.

```python
from openai import OpenAI

# init the client but point it to TGI
client = OpenAI(
    base_url="http://localhost:8080/v1/",
    api_key="-"
)

chat_completion = client.chat.completions.create(
    model="tgi",
    messages=[
        {"role": "system", "content": "You are a helpful assistant." },
        {"role": "user", "content": "What is deep learning?"}
    ],
    stream=True
)

# iterate and print stream
for message in chat_completion:
    print(message)
```

## Preparing Models for TGI

To serve a model with TGI, ensure it meets these requirements:

1. **Supported Architecture**: Verify your model architecture is supported (Llama, BLOOM, T5, etc.)

2. **Model Format**: Convert weights to safetensors format for faster loading:

```python
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("your-model")
state_dict = model.state_dict()
save_file(state_dict, "model.safetensors")
```

3. **Quantization** (optional): Quantize your model to reduce memory usage:

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16"
)

model = AutoModelForCausalLM.from_pretrained(
    "your-model",
    quantization_config=quantization_config
)
```

## References

- [Text Generation Inference Documentation](https://huggingface.co/docs/text-generation-inference)
- [TGI GitHub Repository](https://github.com/huggingface/text-generation-inference)
- [Hugging Face Model Hub](https://huggingface.co/models)
- [TGI API Reference](https://huggingface.co/docs/text-generation-inference/api_reference)