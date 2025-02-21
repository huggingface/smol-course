# Text Generation Inference (TGI)

Text Generation Inference（简称 TGI）是一个由 Hugging Face 开发的工具包，主要用于对大语言模型进行部署和服务搭建。它旨在为常用的开源大语言模型（LLMs）实现高性能文本生成。TGI 被 Hugging Chat 用于实际生产，Hugging Chat 是一个面向开源模型的开源交互界面。


## 为什么使用 TGI？

TGI 解决了在生产环境中部署大语言模型的关键难题。虽然许多框架在模型开发方面表现出色，但 TGI 专门针对生产部署和扩展进行了优化。它的一些关键特性包括：

- **张量并行**：TGI 能够通过张量并行技术将模型分割到多个 GPU 上，这对于高效部署更大的模型至关重要。
- **连续 batch 处理**：连续 batch 处理系统通过动态处理请求，最大限度地提高 GPU 利用率，同时诸如 Flash Attention 和 Paged Attention 等优化方法显著降低了内存使用并提高了速度。
- **token streaming**：实时应用程序受益于通过服务器发送事件（Server-Sent Events）实现的 token streaming 技术，以最低延迟提供响应。

## 怎样使用 TGI

### 基本 Python 用法

TGI 采用了一种简洁而强大的 REST API 集成方式，这使得它能轻松与你的应用程序整合。

### 使用 REST API

TGI 提供了一个接受 JSON 数据的 RESTful API 。这使得任何能够发出 HTTP 请求的编程语言或工具都可以访问它。以下是一个使用 curl 的基本示例：

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

### 使用 `huggingface_hub` 的 Python 客户端

`huggingface_hub` 的 Python 客户端可以处理连接管理、请求格式化以及响应解析。示例如下：


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


### 使用 OpenAI 格式的 API

很多代码库支持 OpenAI API，你也可以用对应的客户端请求去和 TGI 进行互动。

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

## 为 TGI 部署准备模型

如果想借助 TGI 给模型构建服务端，你还需要执行以下步骤：

1. **支持的模型架构**：检查你的模型结构，看看是否受 TGI 支持（目前支持 Llama、BLOOM、T5 等)

2. **模型格式**：将模型权重转换为 safetensors 格式，加速模型参数加载

```python
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("your-model")
state_dict = model.state_dict()
save_file(state_dict, "model.safetensors")
```

1. **量化**（可选步骤）：对模型进行量化，以减少内存使用：

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

## 参考资料

- [Text Generation Inference 官方文档](https://huggingface.co/docs/text-generation-inference)
- [TGI 在 GitHub 上的代码仓库](https://github.com/huggingface/text-generation-inference)
- [Hugging Face Model Hub](https://huggingface.co/models)
- [TGI 的 API 参考文档](https://huggingface.co/docs/text-generation-inference/api_reference)