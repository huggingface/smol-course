# Modelos de bate-papo

Os modelos de bate-papo são essenciais para estruturar interações entre modelos de linguagem e usuários. Eles fornecem um formato consistente para conversas, garantindo que os modelos entendam o contexto e o papel de cada mensagem, enquanto mantêm os padrões de resposta apropriados.

## Modelos Base vs Modelos de Instrução

Um modelo base é treinado em dados de texto bruto para prever o próximo token, enquanto um modelo de instrução é ajustado especificamente para seguir as instruções e se envolver em conversas. Por exemplo, `SmolLM2-135M` é um modelo base, enquanto o `SmolLM2-135M-Instruct` é sua variante ajustada por instrução.

Para fazer um modelo base se comportar como um modelo de instrução, precisamos formatar nossos prompts de uma maneira consistente que o modelo possa entender. É aqui que entram os modelos de bate-papo. ChatML é um desses formatos de modelo que estrutura conversas com indicadores claros de papéis (sistema, usuário, assistente).

É importante notar que o modelo base pode ser ajustado finamente em diferentes modelos de bate-papo, então, quando estamos usando um modelo de instrução, precisamos garantir que estamos usando o modelo de bate-papo correto.

## Entendendo os Modelos de Bate-papo

Na sua essência, os modelos de bate-papo definem como as conversas devem ser formatadas ao se comunicar com um modelo de linguagem. Eles incluem instruções no nível do sistema, mensagens do usuário e respostas assistentes em um formato estruturado que o modelo pode entender. Essa estrutura ajuda a manter a consistência entre as interações e garante que o modelo responda adequadamente a diferentes tipos de input de dados. Abaixo está um exemplo de modelo de bate-papo:

```sh
<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
<|im_start|>assistant
```

O módulo `transformers` cuidará dos modelos de bate-papo para você em relação ao tokenizador do modelo. Leia mais sobre como os transformadores criam modelos de bate-papo [aqui](https://huggingface.co/docs/transformers/en/chat_templating#how-do-i-use-chat-templates). Tudo o que precisamos fazer é estruturar nossas mensagens da maneira correta e o tokenizador cuidará do resto. Abaixo, você verá um exemplo básico de uma conversa:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant focused on technical topics."},
    {"role": "user", "content": "Can you explain what a chat template is?"},
    {"role": "assistant", "content": "A chat template structures conversations between users and AI models..."}
]
```

Vamos ir passo-a-passo no exemplo acima e ver como ele mapeia o formato do modelo de bate-papo.

## Mensagens do Sistema

As mensagens do sistema definem a base de como o modelo deve se comportar. Elas agem como instruções persistentes que influenciam todas as interações subsequentes. Por exemplo:

```python
system_message = {
    "role": "system",
    "content": "You are a professional customer service agent. Always be polite, clear, and helpful."
}
```

## Conversas

Os modelos de bate-papo mantêm o contexto através do histórico de conversas, armazenando conversas anteriores entre os usuários e o assistente. Isso permite que as conversas de multi-turno sejam mais coerentes:

```python
conversation = [
    {"role": "user", "content": "I need help with my order"},
    {"role": "assistant", "content": "I'd be happy to help. Could you provide your order number?"},
    {"role": "user", "content": "It's ORDER-123"},
]
```

## Implementação com Transformadores

O módulo transformers fornece suporte interno para modelos de bate-papo. Veja como usá-los:

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

## Formatação Personalizada

Você pode personalizar como tipos diferentes de mensagens são formatadas. Por exemplo, adicionando tokens especiais ou formatando para diferentes funções:

```python
template = """
<|system|>{system_message}
<|user|>{user_message}
<|assistant|>{assistant_message}
""".lstrip()
```

## Suporte do Multi-turno

Os modelos podem lidar com conversas multi-turno complexas enquanto mantêm o contexto:

```python
messages = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is calculus?"},
    {"role": "assistant", "content": "Calculus is a branch of mathematics..."},
    {"role": "user", "content": "Can you give me an example?"},
]
```

⏭️ [Próxima: Supervised Fine-Tuning (Ajuste Fino Supervisionado)](./supervised_fine_tuning.md)

## Resources

- [Guia de modelos de bate-papo - Hugging Face](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Documentação do módulo Transformers](https://huggingface.co/docs/transformers)
- [Repositório de exemplos de modelos de bate-papo](https://github.com/chujiezheng/chat_templates) 
