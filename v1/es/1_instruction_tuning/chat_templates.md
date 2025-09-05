# Plantillas de Chat

Las plantillas de chat son esenciales para estructurar las interacciones entre modelos de lenguaje y usuarios. Proporcionan un formato consistente para las conversaciones, asegurando que los modelos comprendan el contexto y el rol de cada mensaje, manteniendo patrones de respuesta apropiados.

## Modelos Base vs Modelos Instructivos

Un modelo base se entrena con datos de texto crudo para predecir el siguiente token, mientras que un modelo instructivo se ajusta específicamente para seguir instrucciones y participar en conversaciones. Por ejemplo, `SmolLM2-135M` es un modelo base, mientras que `SmolLM2-135M-Instruct` es su variante ajustada a instrucciones.

Para hacer que un modelo base se comporte como un modelo instructivo, necesitamos formatear nuestros prompts de manera consistente para que el modelo los entienda. Aquí es donde entran las plantillas de chat. ChatML es un formato de plantilla que estructura las conversaciones con indicadores de rol claros (sistema, usuario, asistente).

Es importante notar que un modelo base podría ser ajustado con diferentes plantillas de chat, por lo que, al usar un modelo instructivo, debemos asegurarnos de estar utilizando la plantilla de chat correcta.

## Comprendiendo las Plantillas de Chat

En su núcleo, las plantillas de chat definen cómo se deben formatear las conversaciones al comunicarse con un modelo de lenguaje. Incluyen instrucciones a nivel de sistema, mensajes del usuario y respuestas del asistente en un formato estructurado que el modelo puede entender. Esta estructura ayuda a mantener la coherencia en las interacciones y asegura que el modelo responda adecuadamente a diferentes tipos de entradas. A continuación, se muestra un ejemplo de una plantilla de chat:

```sh
<|im_start|>user
¡Hola!<|im_end|>
<|im_start|>assistant
¡Mucho gusto!<|im_end|>
<|im_start|>user
¿Puedo hacer una pregunta?<|im_end|>
<|im_start|>assistant
``` 

La librería `transformers` se encarga de las plantillas de chat por ti en relación con el tokenizador del modelo. Lee más sobre cómo `transformers` construye las plantillas de chat [aquí](https://huggingface.co/docs/transformers/en/chat_templating#how-do-i-use-chat-templates). Todo lo que tenemos que hacer es estructurar nuestros mensajes de la forma correcta y el tokenizador se encargará del resto. Aquí tienes un ejemplo básico de una conversación:

```python
messages = [
    {"role": "system", "content": "Eres un asistente útil centrado en temas técnicos."},
    {"role": "user", "content": "¿Puedes explicar qué es una plantilla de chat?"},
    {"role": "assistant", "content": "Una plantilla de chat estructura las conversaciones entre los usuarios y los modelos de IA..."}
]
```

Vamos a desglosar el ejemplo anterior y ver cómo se mapea al formato de plantilla de chat.

## Mensajes del Sistema

Los mensajes del sistema establecen la base para el comportamiento del modelo. Actúan como instrucciones persistentes que influyen en todas las interacciones posteriores. Por ejemplo:

```python
system_message = {
    "role": "system",
    "content": "Eres un agente de atención al cliente profesional. Siempre sé educado, claro y útil."
}
```

## Conversaciones

Las plantillas de chat mantienen el contexto a través del historial de la conversación, almacenando intercambios previos entre el usuario y el asistente. Esto permite conversaciones más coherentes en múltiples turnos:

```python
conversation = [
    {"role": "user", "content": "Necesito ayuda con mi pedido"},
    {"role": "assistant", "content": "Estaré encantado de ayudarte. ¿Podrías proporcionarme tu número de pedido?"},
    {"role": "user", "content": "Es el PEDIDO-123"},
]
```

## Implementación con Transformers

La librería `transformers` proporciona soporte integrado para plantillas de chat. Aquí te mostramos cómo usarlas:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

messages = [
    {"role": "system", "content": "Eres un asistente útil para programación."},
    {"role": "user", "content": "Escribe una función en Python para ordenar una lista"},
]

# Aplica la plantilla de chat
formatted_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

## Formato Personalizado

Puedes personalizar cómo se formatean los diferentes tipos de mensajes. Por ejemplo, añadiendo tokens especiales o formato para diferentes roles:

```python
template = """
<|system|>{system_message}
<|user|>{user_message}
<|assistant|>{assistant_message}
""".lstrip()
```

## Soporte para Conversaciones de Varios Turnos

Las plantillas pueden manejar conversaciones complejas de varios turnos mientras mantienen el contexto:

```python
messages = [
    {"role": "system", "content": "Eres un tutor de matemáticas."},
    {"role": "user", "content": "¿Qué es el cálculo?"},
    {"role": "assistant", "content": "El cálculo es una rama de las matemáticas..."},
    {"role": "user", "content": "¿Puedes darme un ejemplo?"},
]
```

⏭️ [Siguiente: Fine-tuning Supervisado](./supervised_fine_tuning.md)

## Recursos

- [Guía de Plantillas de Chat de Hugging Face](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Documentación de Transformers](https://huggingface.co/docs/transformers)
- [Repositorio de Ejemplos de Plantillas de Chat](https://github.com/chujiezheng/chat_templates)
```