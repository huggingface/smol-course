# Định dạng Chat

Định dạng Chat (Chat templates) là yếu tố thiết yếu trong cấu trúc các tương tác giữa mô hình ngôn ngữ và người dùng. Chúng cung cấp một định dạng nhất quán cho các cuộc hội thoại, đảm bảo rằng các mô hình hiểu được ngữ cảnh và vai trò của mỗi tin nhắn trong khi duy trì các mẫu phản hồi phù hợp.

## Mô hình gốc (Base Models) và Mô hình chỉ thị (Instruct Models)

Mô hình gốc được huấn luyện trên dữ liệu văn bản thô để dự đoán *token* tiếp theo, trong khi Mô hình chỉ thị được tiếp tục tinh chỉnh đặc biệt để tuân theo chỉ thị và tham gia vào hội thoại. Ví dụ, `SmolLM2-135M` là một mô hình gốc, trong khi `SmolLM2-135M-Instruct` là phiên bản đã được điều chỉnh.

Để làm cho Mô hình gốc hoạt động như một Mô hình chỉ thị, chúng ta cần *định dạng prompt* của mình theo cách nhất quán mà mô hình có thể hiểu được. Đây là lúc *định dạng chat* phát huy tác dụng. **ChatML** là một định dạng template như vậy, với cấu trúc các cuộc hội thoại có chỉ định vai trò rõ ràng (system, user, assistant).

Điều quan trọng cần lưu ý là một Mô hình gốc có thể được huấn luyện với các *định dạng chat* khác nhau, vì vậy khi chúng ta sử dụng một Mô hình chỉ thị, chúng ta cần đảm bảo đang sử dụng đúng *định dạng chat*.

## Tìm hiểu về Định dạng Chat

Về cốt lõi, *định dạng chat* định nghĩa cách các cuộc hội thoại nên được định dạng khi giao tiếp với một mô hình ngôn ngữ. Chúng bao gồm các hướng dẫn hệ thống (system), tin nhắn người dùng (user) và phản hồi của trợ lý (assistant) trong một định dạng có cấu trúc mà mô hình có thể hiểu được. Cấu trúc này giúp duy trì tính nhất quán trong các tương tác và đảm bảo mô hình phản hồi phù hợp với các loại đầu vào khác nhau. Dưới đây là một ví dụ về chat template:

```sh
<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
<|im_start|>assistant
```

Thư viện `transformers` sẽ xử lý *định dạng chat* cho bạn liên quan đến tokenizer của mô hình. Đọc thêm về cách transformers xây dựng *định dạng chat* [tại đây](https://huggingface.co/docs/transformers/en/chat_templating#how-do-i-use-chat-templates). Tất cả những gì chúng ta cần làm là cấu trúc tin nhắn của mình theo cách chính xác và *tokenizer() sẽ xử lý phần còn lại. Đây là một ví dụ cơ bản về một cuộc hội thoại:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant focused on technical topics."},
    {"role": "user", "content": "Can you explain what a chat template is?"},
    {"role": "assistant", "content": "A chat template structures conversations between users and AI models..."}
]
```

Hãy cùng nhau phân tích ví dụ trên để hiểu hơn về *định dạng chat*

## Mệnh lệnh hệ thống (System Prompt)

Mệnh lệnh hệ thống thiết lập nền tảng cho cách mô hình nên hoạt động. Chúng đóng vai trò như các hướng dẫn ảnh hưởng liên tục đến tất cả các tương tác tiếp theo. Ví dụ:

```python
system_message = {
    "role": "system",
    "content": "You are a professional customer service agent. Always be polite, clear, and helpful."
}
```

## Cuộc hội thoại

*Định dạng chat* duy trì ngữ cảnh thông qua lịch sử hội thoại, lưu trữ các trao đổi trước đó giữa người dùng và trợ lý. Điều này cho phép các cuộc hội thoại mạch lạc hơn:

```python
conversation = [
    {"role": "user", "content": "I need help with my order"},
    {"role": "assistant", "content": "I'd be happy to help. Could you provide your order number?"},
    {"role": "user", "content": "It's ORDER-123"},
]
```

## Triển khai với thư viện Transformers

Thư viện transformers cung cấp hỗ trợ tích hợp cho *định dạng chat*. Đây là cách sử dụng:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Write a Python function to sort a list"},
]

# Apply the chat template
formatted_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

## Tuỳ chỉnh Định dạng Chat
Bạn có thể tùy chỉnh cách định dạng các loại tin nhắn khác nhau. Ví dụ, thêm *special token* hoặc định dạng cho các vai trò khác nhau:

```python
template = """
<|system|>{system_message}
<|user|>{user_message}
<|assistant|>{assistant_message}
""".lstrip()
```

## Hỗ trợ hội thoại đa lượt (multi-turn conversations)

Với *định dạng chat*, mô hình có thể xử lý các cuộc hội thoại phức tạp nhiều lượt trong khi vẫn duy trì ngữ cảnh:

```python
messages = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is calculus?"},
    {"role": "assistant", "content": "Calculus is a branch of mathematics..."},
    {"role": "user", "content": "Can you give me an example?"},
]
```

⏭️ [Tiếp theo: Huấn luyện có giám sát](./supervised_fine_tuning.md)

## Tài liệu tham khảo

- [Hướng dẫn Định dạng Chat của Hugging Face](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Tài liệu về thư viện Transformers](https://huggingface.co/docs/transformers)
- [Ví dụ về Định dạng Chat](https://github.com/chujiezheng/chat_templates) 
