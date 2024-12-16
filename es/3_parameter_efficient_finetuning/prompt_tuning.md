# Ajuste de Prompts

El ajuste de prompts es un enfoque eficiente en términos de parámetros que modifica las representaciones de entrada en lugar de los pesos del modelo. A diferencia del fine-tuning tradicional que actualiza todos los parámetros del modelo, el ajuste de prompts agrega y optimiza un pequeño conjunto de tokens entrenables mientras mantiene el modelo base congelado.

## Entendiendo el Ajuste de Prompts

El ajuste de prompts es una alternativa eficiente en términos de parámetros al fine-tuning del modelo que antepone vectores continuos entrenables (prompts suaves) al texto de entrada. A diferencia de los prompts discretos de texto, estos prompts suaves se aprenden mediante retropropagación mientras el modelo de lenguaje se mantiene congelado. El método fue introducido en ["El poder de la escala para el ajuste de prompts eficiente en parámetros"](https://arxiv.org/abs/2104.08691) (Lester et al., 2021), que demostró que el ajuste de prompts se vuelve más competitivo con el fine-tuning del modelo a medida que aumenta el tamaño del modelo. En el artículo, alrededor de 10 mil millones de parámetros, el ajuste de prompts iguala el rendimiento del fine-tuning del modelo mientras solo modifica unos pocos cientos de parámetros por tarea.

Estos prompts suaves son vectores continuos en el espacio de incrustación del modelo que se optimizan durante el entrenamiento. A diferencia de los prompts discretos tradicionales que usan tokens de lenguaje natural, los prompts suaves no tienen un significado inherente, pero aprenden a provocar el comportamiento deseado del modelo congelado mediante el descenso de gradiente. La técnica es particularmente efectiva para escenarios de múltiples tareas, ya que cada tarea requiere almacenar solo un pequeño vector de prompt (normalmente unos pocos cientos de parámetros) en lugar de una copia completa del modelo. Este enfoque no solo mantiene una huella de memoria mínima, sino que también permite una adaptación rápida de tareas simplemente intercambiando los vectores de prompt sin necesidad de recargar el modelo.

## Proceso de Entrenamiento

Los prompts suaves suelen ser entre 8 y 32 tokens y se pueden inicializar de forma aleatoria o a partir de texto existente. El método de inicialización juega un papel crucial en el proceso de entrenamiento, con la inicialización basada en texto a menudo obteniendo mejores resultados que la aleatoria.

Durante el entrenamiento, solo se actualizan los parámetros del prompt mientras que el modelo base permanece congelado. Este enfoque centrado utiliza objetivos de entrenamiento estándar, pero requiere una atención cuidadosa a la tasa de aprendizaje y el comportamiento de los gradientes de los tokens del prompt.

## Implementación con PEFT

La libreria PEFT facilita la implementación del ajuste de prompts. Aquí tienes un ejemplo básico:

```python
from peft import PromptTuningConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Carga el modelo base
model = AutoModelForCausalLM.from_pretrained("your-base-model")
tokenizer = AutoTokenizer.from_pretrained("your-base-model")

# Configura el ajuste de prompt
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=8,  # Número de tokens entrenables
    prompt_tuning_init="TEXT",  # Inicializa a partir de texto
    prompt_tuning_init_text="Clasifique si este texto es positivo o negativo:",
    tokenizer_name_or_path="your-base-model",
)

# Crea el modelo ajustable por prompt
model = get_peft_model(model, peft_config)
```

## Comparación con Otros Métodos

Cuando se compara con otros enfoques de PEFT, el ajuste de prompts destaca por su eficiencia. Mientras que LoRA ofrece bajos recuentos de parámetros y uso de memoria pero requiere cargar adaptadores para el cambio de tareas, el ajuste de prompts logra un uso de recursos aún menor y permite un cambio de tareas inmediato. El fine-tuning completo, en contraste, exige recursos significativos y requiere copias separadas del modelo para diferentes tareas.

| Método            | Parámetros  | Memoria  | Cambio de Tarea |
|-------------------|-------------|----------|-----------------|
| Ajuste de Prompts | Muy Bajo    | Mínimo   | Fácil           |
| LoRA              | Bajo        | Bajo     | Requiere Carga  |
| Fine-tuning Completo   | Alto        | Alto     | Nueva Copia     |

Al implementar el ajuste de prompts, comienza con un pequeño número de tokens virtuales (8-16) y aumenta solo si la complejidad de la tarea lo requiere. La inicialización con texto generalmente da mejores resultados que la aleatoria, especialmente cuando se utiliza texto relevante para la tarea. La estrategia de inicialización debe reflejar la complejidad de la tarea objetivo.

El entrenamiento requiere consideraciones ligeramente diferentes al fine-tuning completo. Las tasas de aprendizaje más altas suelen funcionar bien, pero es esencial monitorear cuidadosamente los gradientes de los tokens de prompt. La validación regular con ejemplos diversos ayuda a garantizar un rendimiento robusto en diferentes escenarios.

## Aplicaciones

El ajuste de prompts sobresale en varios escenarios:

1. Despliegue de múltiples tareas
2. Entornos con limitaciones de recursos
3. Adaptación rápida de tareas
4. Aplicaciones sensibles a la privacidad

A medida que los modelos se hacen más pequeños, el ajuste de prompts se vuelve menos competitivo en comparación con el fine-tuning completo. Por ejemplo, en modelos como SmolLM2, el ajuste de prompts es menos relevante que el fine-tuning completo.

## Próximos Pasos

⏭️ Pasa al [Tutorial de Adaptadores LoRA](./notebooks/finetune_sft_peft.ipynb) para aprender cómo ajustar un modelo con adaptadores LoRA.

## Recursos
- [Documentación de PEFT](https://huggingface.co/docs/peft)
- [Papel sobre Ajuste de Prompts](https://arxiv.org/abs/2104.08691)
- [Recetario de Hugging Face](https://huggingface.co/learn/cookbook/prompt_tuning_peft)