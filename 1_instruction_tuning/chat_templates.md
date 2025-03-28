# 聊天模板

如果想将模型与用户的交互信息结构化，那么一个聊天模板（chat template）就是必需的。它为对话提供了一个固定格式，让模型能够知道上下文信息以及每条消息是由谁发出的，只有这样模型才能生成恰当的回答。

## 基础模型 vs 指令模型

基础模型（base model）指的是在未经整理的文本数据上训练、用于预测下一个 token 的模型，而指令模型（instruct model）则是通过微调来跟随指令、参与对话的模型。举例来说，`SmolLM2-135M` 就是基础模型，而 `SmolLM2-135M-Instruct` 则是前者经指令调优得到的指令模型。

为了让基础模型成为指令模型，我们需要对我们的输入提示词进行规范化，用一种固定的格式输入给模型，以便于模型理解。这就用到**聊天模板**了。举例来说，ChatML 就是一个这样的模板，它将对话过程完全结构化，清晰地指明了每段信息是由哪个角色（系统、用户、助手）说出的。

需要注意，一个基础模型可以往不同的聊天模板上微调。所以当我们使用训练好的指令模型时，我们也需要注意不要用错聊天模板。

## 聊天模板简介

聊天模板定义了当用户和语言模型对话时，对话信息应该遵循什么样的格式。这其中包含来自三个角色的信息：系统级的指令、用户发出的信息、AI 助手的回答。这使得每次人机交互的信息格式都是一致的，确保模型针对不同问题都能恰当回答。下面就是一个聊天模板示例：

```sh
<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
<|im_start|>assistant
```

实际上，如果你使用 `transformers` 库的 tokenizer，它将会为我们将对话信息转化为聊天模板形式。你可以在[这里](https://huggingface.co/docs/transformers/en/chat_templating#how-do-i-use-chat-templates)查看相关文档。我们仅需将对话信息结构化，后面的事情交给 tokenizer 即可。比如，你可以把聊天信息写成这样：

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant focused on technical topics."},
    {"role": "user", "content": "Can you explain what a chat template is?"},
    {"role": "assistant", "content": "A chat template structures conversations between users and AI models..."}
]
```

接下来，我们将分解聊天信息的组成：系统信息和对话部分。

### 系统消息

系统消息从基本层面定义了模型应有的行为。它会影响接下来所有交互。看下列示例就能明白：

```python
system_message = {
    "role": "system",
    "content": "You are a professional customer service agent. Always be polite, clear, and helpful."
}
```

### 对话部分

聊天模板也需要保留对话历史记录，将之前发生的人机对话保存下来，作为后续对话的参考。只有这样，我们才能实现多轮交互式对话。

```python
conversation = [
    {"role": "user", "content": "I need help with my order"},
    {"role": "assistant", "content": "I'd be happy to help. Could you provide your order number?"},
    {"role": "user", "content": "It's ORDER-123"},
]
```

## 使用 Transformers 构建聊天模板 

使用 `transformers` 构建聊天模板的示例如下： 

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

上述代码运行完后，`formatted_chat` 应该是这样：
```
'<|im_start|>system\nYou are a helpful coding assistant.<|im_end|>\n<|im_start|>user\nWrite a Python function to sort a list<|im_end|>\n<|im_start|>assistant\n'
```

### 自定义聊天模板格式

你也可以自定义聊天模板格式，比如为不同角色的信息添加特殊的 token 来作为标识：

```python
template = """
<|system|>{system_message}
<|user|>{user_message}
<|assistant|>{assistant_message}
""".lstrip()
```

### 对多轮对话的支持

聊天模板可以处理复杂多轮对话，同时保留上下文信息：

```python
messages = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is calculus?"},
    {"role": "assistant", "content": "Calculus is a branch of mathematics..."},
    {"role": "user", "content": "Can you give me an example?"},
]
```

⏭️ [下一节课程：有监督微调](./supervised_fine_tuning.md)

## 其它学习资源

- [Hugging Face 聊天模板使用指南](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Transformers 官方文档](https://huggingface.co/docs/transformers)
- [包含各种聊天模板的代码仓库](https://github.com/chujiezheng/chat_templates) 
