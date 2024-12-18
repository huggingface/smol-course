# 대화 템플릿(Chat Templates)

대화 템플릿은 언어 모델과 사용자 간 상호작용을 구조화하는 데 필수적입니다. 이 템플릿은 대화에 일관된 형식을 제공하여, 모델이 각 메시지의 맥락과 역할을 이해하고 적절한 응답 패턴을 유지할 수 있도록 합니다.

## 기본 모델 vs 지시 모델

기본 모델은 다음 토큰을 예측하기 위해 대량의 원시 텍스트 데이터로 학습되는 반면, 지시 모델은 특정 지시를 따르고 대화를 나눌 수 있도록 미세 조정된 모델입니다. 예를 들어, `SmolLM2-135M`은 기본 모델이고, `SmolLM2-135M-Instruct`는 이를 지시 조정한 변형 모델입니다.

기본 모델이 지시 모델처럼 동작하도록 만들기 위해서는 모델이 이해할 수 있는 방식으로 프롬프트를 일관되게 만들어야 합니다. 이때 대화 템플릿이 유용합니다. ChatML은 이러한 템플릿 형식 중 하나로 대화를 구조화하고 명확한 역할 지시자(시스템, 사용자, 어시스턴트)를 제공합니다.

기본 모델은 서로 다른 대화 템플릿으로 미세 조정될 수 있으므로, 지시 모델을 사용할 때는 반드시 해당 모델에 적합한 대화 템플릿을 사용하는 것이 중요합니다.

## 대화 템플릿 이해하기

대화 템플릿의 핵심은 언어 모델과 소통할 때 대화가 어떤 형식으로 이루어져야 하는지 정의하는 것입니다. 템플릿에는 시스템 수준의 지침, 사용자 메시지, 어시스턴트의 응답이 포함되어 있으며, 모델이 이해할 수 있는 구조화된 형식을 제공합니다. 이러한 구조는 상호작용의 일관성을 유지하고, 모델이 다양한 유형의 입력에 적절히 응답하도록 돕습니다. 아래는 채팅 템플릿의 예시입니다:

```sh
<|im_start|>user
안녕하세요!<|im_end|>
<|im_start|>assistant
만나서 반갑습니다!<|im_end|>
<|im_start|>user
질문을 해도 될까요?<|im_end|>
<|im_start|>assistant
```

`transformers` 라이브러리는 모델의 토크나이저와 관련하여 대화 템플릿을 자동으로 처리해줍니다. 대화 템플릿이 transformers에서 어떻게 구성되는지 자세히 알아보려면 [여기](https://huggingface.co/docs/transformers/en/chat_templating#how-do-i-use-chat-templates)를 참고하세요. 우리가 메시지를 올바른 형식으로 구조화하기만 하면 나머지는 토크나이저가 처리합니다. 아래는 기본적인 대화 예시입니다:
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant focused on technical topics."},
    {"role": "user", "content": "Can you explain what a chat template is?"},
    {"role": "assistant", "content": "A chat template structures conversations between users and AI models..."}
]
```

위 예시를 자세히 보면서 대화 템플릿 형식에 어떻게 매핑되는지 살펴봅시다.

## 시스템 메시지

시스템 메시지는 모델의 기본 동작 방식을 설정합니다. 이는 이후 모든 상호작용에 영향을 미치는 지속적인 지침이 됩니다.

```python
system_message = {
    "role": "system",
    "content": "You are a professional customer service agent. Always be polite, clear, and helpful."
}
```

## 대화

대화 템플릿은 대화 기록을 통해 맥락을 유지하며, 사용자와 어시스턴트 간의 이전 대화를 저장합니다. 이를 통해 더욱 일관된 멀티 턴 대화가 가능해집니다:

```python
conversation = [
    {"role": "user", "content": "I need help with my order"},
    {"role": "assistant", "content": "I'd be happy to help. Could you provide your order number?"},
    {"role": "user", "content": "It's ORDER-123"},
]
```

## Transformers 기반 구현

transformers 라이브러리는 대화 템플릿을 위한 내장 기능을 지원합니다.사용 방법은 다음과 같습니다:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Write a Python function to sort a list"},
]

# 대화 템플릿 적용
formatted_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

## 사용자 정의 형식

메시지 유형별로 원하는 형식을 설정할 수 있습니다. 예를 들어, 역할에 따라 특수 토큰을 추가하거나 형식을 지정할 수 있습니다:

```python
template = """
<|system|>{system_message}
<|user|>{user_message}
<|assistant|>{assistant_message}
""".lstrip()
```

## 멀티 턴 지원

템플릿은 문맥을 유지하면서 복잡한 멀티 턴 대화도 처리할 수 있습니다:

```python
messages = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is calculus?"},
    {"role": "assistant", "content": "Calculus is a branch of mathematics..."},
    {"role": "user", "content": "Can you give me an example?"},
]
```

⏭️ [다음: Supervised Fine-Tuning](./supervised_fine_tuning.md)

## 참고

- [Hugging Face Chat Templating Guide](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Chat Templates Examples Repository](https://github.com/chujiezheng/chat_templates) 
