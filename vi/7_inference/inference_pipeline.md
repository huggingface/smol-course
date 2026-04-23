# Suy luận cơ bản với Pipeline Transformers

Abstraction `pipeline` trong 🤗 Transformers cung cấp một cách đơn giản để thực hiện suy luận với bất kỳ mô hình nào từ Hugging Face Hub. Nó xử lý tất cả các bước tiền xử lý và hậu xử lý, giúp người dùng dễ dàng sử dụng mô hình mà không cần kiến thức sâu về kiến trúc hay yêu cầu của chúng.

## Cách hoạt động của Pipelines

Pipelines của Hugging Face đơn giản hóa quy trình học máy bằng cách tự động hóa ba giai đoạn quan trọng giữa đầu vào thô và đầu ra có thể đọc được:

**Giai đoạn Tiền xử lý**
Pipeline sẽ chuẩn bị các đầu vào thô cho mô hình. Điều này thay đổi tùy thuộc vào loại đầu vào:
- Đầu vào văn bản trải qua quá trình phân tách (tokenization) để chuyển các từ thành các ID token phù hợp với mô hình
- Hình ảnh được thay đổi kích thước và chuẩn hóa để phù hợp với yêu cầu của mô hình
- Âm thanh được xử lý thông qua việc trích xuất tính năng để tạo ra phổ tần (spectrograms) hoặc các biểu diễn khác

**Suy luận Mô hình**
Trong quá trình chạy mô hình, pipeline:
- Tự động xử lý batching các đầu vào để tăng hiệu quả
- Đặt tính toán lên thiết bị tối ưu (CPU/GPU)
- Áp dụng các tối ưu hóa hiệu suất như suy luận với độ chính xác nửa (FP16) khi được hỗ trợ

**Giai đoạn Hậu xử lý**
Cuối cùng, pipeline chuyển đổi đầu ra thô từ mô hình thành kết quả hữu ích:
- Giải mã các ID token trở lại thành văn bản có thể đọc được
- Biến đổi các logits thành điểm xác suất
- Định dạng đầu ra theo tác vụ cụ thể (ví dụ: nhãn phân loại, văn bản sinh ra)

Abstraction này giúp bạn tập trung vào logic ứng dụng trong khi pipeline xử lý sự phức tạp kỹ thuật của suy luận mô hình.

## Cách sử dụng cơ bản

Dưới đây là cách sử dụng pipeline để tạo văn bản:

```python
from transformers import pipeline

# Tạo pipeline với một mô hình cụ thể
generator = pipeline(
    "text-generation",
    model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

# Tạo văn bản
response = generator(
    "Write a short poem about coding:",
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7
)
print(response[0]['generated_text'])
```

## Các tùy chọn cấu hình chính

### Tải mô hình
```python
# Suy luận trên CPU
generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-1.7B-Instruct", device="cpu")

# Suy luận trên GPU
generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-1.7B-Instruct", device=0)

# Đặt thiết bị tự động
generator = pipeline(
    "text-generation",
    model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    device_map="auto",
    torch_dtype="auto"
)
```

### Tham số tạo văn bản

```python
response = generator(
    "Translate this to French:",
    max_new_tokens=100,     # Độ dài tối đa của văn bản sinh ra
    do_sample=True,         # Sử dụng sampling thay vì greedy decoding
    temperature=0.7,        # Điều chỉnh độ ngẫu nhiên (cao hơn = ngẫu nhiên hơn)
    top_k=50,               # Giới hạn đến top k token
    top_p=0.95,             # Ngưỡng sampling hạt nhân
    num_return_sequences=1  # Số lượng văn bản sinh ra khác nhau
)
```

## Xử lý nhiều đầu vào

Pipelines có thể xử lý nhiều đầu vào một cách hiệu quả thông qua batching:

```python
# Chuẩn bị nhiều prompt
prompts = [
    "Write a haiku about programming:",
    "Explain what an API is:",
    "Write a short story about a robot:"
]

# Xử lý tất cả các prompt một cách hiệu quả
responses = generator(
    prompts,
    batch_size=4,              # Số lượng prompt xử lý cùng một lúc
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7
)

# In kết quả
for prompt, response in zip(prompts, responses):
    print(f"Prompt: {prompt}")
    print(f"Response: {response[0]['generated_text']}\n")
```

## Tích hợp với Web Server

Dưới đây là cách tích hợp pipeline vào một ứng dụng FastAPI:

```python
from fastapi import FastAPI, HTTPException
from transformers import pipeline
import uvicorn

app = FastAPI()

# Khởi tạo pipeline toàn cục
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

## Hạn chế

Mặc dù pipelines rất hữu ích cho việc tạo mẫu và triển khai quy mô nhỏ, chúng có một số hạn chế:

- Các tùy chọn tối ưu hóa hạn chế so với các giải pháp phục vụ chuyên dụng
- Không hỗ trợ các tính năng nâng cao như batching động
- Có thể không phù hợp cho các khối lượng công việc sản xuất yêu cầu tốc độ cao

Đối với các triển khai sản xuất với yêu cầu throughput cao, bạn có thể xem xét sử dụng Suy luận Sinh văn Bằng Văn bản (TGI) hoặc các giải pháp phục vụ chuyên biệt khác.

## Tài nguyên

- [Hướng dẫn Pipeline Hugging Face.](https://huggingface.co/docs/transformers/en/pipeline_tutorial)
- [Tham khảo API Pipeline.](https://huggingface.co/docs/transformers/en/main_classes/pipelines)
- [Tham số Sinh văn bản.](https://huggingface.co/docs/transformers/en/main_classes/text_generation)
- [Hướng dẫn Quantization mô hình.](https://huggingface.co/docs/transformers/en/perf_infer_gpu_one)