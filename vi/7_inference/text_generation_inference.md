# Suy luận Sinh Văn Bằng Văn bản (TGI)

Suy luận Sinh Văn Bằng Văn bản (TGI) là một bộ công cụ được phát triển bởi Hugging Face để triển khai và phục vụ các Mô hình Ngôn ngữ Lớn (LLMs). TGI được thiết kế để tối ưu hóa hiệu suất sinh văn cho các mô hình LLM mã nguồn mở phổ biến. TGI đang được sử dụng trong sản xuất bởi Hugging Chat - Giao diện mã nguồn mở cho các mô hình truy cập mở.

## Tại sao nên sử dụng Suy luận Sinh Văn Bằng Văn bản?

Suy luận Sinh Văn Bằng Văn bản giải quyết các thách thức chính trong việc triển khai các mô hình ngôn ngữ lớn vào sản xuất. Trong khi nhiều framework xuất sắc trong việc phát triển mô hình, TGI tối ưu hóa cho việc triển khai sản xuất và mở rộng quy mô. Một số tính năng chính bao gồm:

- **Tensor Parallelism**: TGI có thể phân chia các mô hình trên nhiều GPU thông qua tensor parallelism, điều này rất quan trọng để phục vụ các mô hình lớn một cách hiệu quả.
- **Continuous Batching**: Hệ thống batching liên tục tối đa hóa việc sử dụng GPU bằng cách xử lý các yêu cầu một cách động, trong khi các tối ưu hóa như Flash Attention và Paged Attention giúp giảm đáng kể việc sử dụng bộ nhớ và tăng tốc độ.
- **Token Streaming**: Các ứng dụng thời gian thực được hưởng lợi từ token streaming thông qua Server-Sent Events, cung cấp phản hồi với độ trễ tối thiểu.

## Cách sử dụng Suy luận Sinh Văn Bằng Văn bản

### Sử dụng Python cơ bản

TGI sử dụng một API REST đơn giản nhưng mạnh mẽ, giúp dễ dàng tích hợp vào các ứng dụng của bạn.

### Sử dụng REST API

TGI cung cấp một API RESTful nhận các payload dưới dạng JSON. Điều này giúp API có thể truy cập từ bất kỳ ngôn ngữ lập trình hoặc công cụ nào có thể gửi yêu cầu HTTP. Dưới đây là ví dụ cơ bản sử dụng curl:

```bash
# Yêu cầu sinh văn bản cơ bản
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

### Sử dụng `huggingface_hub` Python Client

Python client `huggingface_hub` giúp quản lý kết nối, định dạng yêu cầu và phân tích phản hồi. Đây là cách bắt đầu sử dụng:

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

### Sử dụng OpenAI API

Nhiều thư viện hỗ trợ API OpenAI, vì vậy bạn có thể sử dụng cùng một client để tương tác với TGI.

```python
from openai import OpenAI

# Khởi tạo client và trỏ tới TGI
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

# Duyệt qua và in stream
for message in chat_completion:
    print(message)
```

## Chuẩn bị Mô hình cho TGI

Để phục vụ mô hình với TGI, hãy đảm bảo mô hình của bạn đáp ứng các yêu cầu sau:

1. **Kiến trúc được hỗ trợ**: Xác minh mô hình của bạn có kiến trúc được hỗ trợ (Llama, BLOOM, T5, v.v.)

2. **Định dạng mô hình**: Chuyển đổi trọng số mô hình sang định dạng safetensors để tải nhanh hơn:

```python
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("your-model")
state_dict = model.state_dict()
save_file(state_dict, "model.safetensors")
```

3. **Quantization** (tùy chọn): Quantize mô hình của bạn để giảm việc sử dụng bộ nhớ:

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

## Tài nguyên tham khảo

- [Tài liệu Suy luận Sinh Văn Bằng Văn bản](https://huggingface.co/docs/text-generation-inference)
- [Kho lưu trữ TGI trên GitHub](https://github.com/huggingface/text-generation-inference)
- [Model Hub của Hugging Face](https://huggingface.co/models)
- [Tham khảo API TGI](https://huggingface.co/docs/text-generation-inference/api_reference)