# Chat Templates

Chat templates are essential for structuring interactions between language models and users. They provide a consistent format for conversations, ensuring that models understand the context and role of each message while maintaining appropriate response patterns.

## Base Models vs Instruct Models

A base model is trained on raw text data to predict the next token, while an instruct model is fine-tuned specifically to follow instructions and engage in conversations. For example, `SmolLM2-135M` is a base model, while `SmolLM2-135M-Instruct` is its instruction-tuned variant.

To make a base model behave like an instruct model, we need to format our prompts in a consistent way that the model can understand. This is where chat templates come in. ChatML is one such template format that structures conversations with clear role indicators (system, user, assistant).

It's important to note that a base model could be fine-tuned on different chat templates, so when we're using an instruct model we need to make sure we're using the correct chat template.

## Understanding Chat Templates

At their core, chat templates define how conversations should be formatted when communicating with a language model. They include system-level instructions, user messages, and assistant responses in a structured format that the model can understand. This structure helps maintain consistency across interactions and ensures the model responds appropriately to different types of inputs. Below is an example of a chat template:

```sh
<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
<|im_start|>assistant
```

The `transformers` library will take care of chat templates for you in relation to the model's tokenizer. Read more about how transformers builds chat templates [here](https://huggingface.co/docs/transformers/en/chat_templating#how-do-i-use-chat-templates). All we have to do is structure our messages in the correct way and the tokenizer will take care of the rest. Here's a basic example of a conversation:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant focused on technical topics."},
    {"role": "user", "content": "Can you explain what a chat template is?"},
    {"role": "assistant", "content": "A chat template structures conversations between users and AI models..."}
]
```

Let's break down the above example, and see how it maps to the chat template format.

## System Messages

System messages set the foundation for how the model should behave. They act as persistent instructions that influence all subsequent interactions. For example:

```python
system_message = {
    "role": "system",
    "content": "You are a professional customer service agent. Always be polite, clear, and helpful."
}
```

## Conversations

Chat templates maintain context through conversation history, storing previous exchanges between users and the assistant. This allows for more coherent multi-turn conversations:

```python
conversation = [
    {"role": "user", "content": "I need help with my order"},
    {"role": "assistant", "content": "I'd be happy to help. Could you provide your order number?"},
    {"role": "user", "content": "It's ORDER-123"},
]
```

## Implementation with Transformers

The transformers library provides built-in support for chat templates. Here's how to use them:

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

## Custom Formatting
You can customize how different message types are formatted. For example, adding special tokens or formatting for different roles:

```python
template = """
<|system|>{system_message}
<|user|>{user_message}
<|assistant|>{assistant_message}
""".lstrip()
```

## Multi-Turn Support

Templates can handle complex multi-turn conversations while maintaining context:

```python
messages = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is calculus?"},
    {"role": "assistant", "content": "Calculus is a branch of mathematics..."},
    {"role": "user", "content": "Can you give me an example?"},
]
```

⏭️ [Next: Supervised Fine-Tuning](./supervised_fine_tuning.md)

## Resources

- [Hugging Face Chat Templating Guide](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Chat Templates Examples Repository](https://github.com/chujiezheng/chat_templates) 
