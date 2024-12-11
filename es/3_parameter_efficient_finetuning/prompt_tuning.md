# Ajuste de Prompts y Prefijos

El ajuste de prompts es un enfoque eficiente en cuanto a parámetros que modifica las representaciones de entrada en lugar de los pesos del modelo. A diferencia del ajuste tradicional, que actualiza todos los parámetros del modelo, el ajuste de prompts agrega y optimiza un pequeño conjunto de tokens entrenables mientras mantiene el modelo base congelado.

## Entendiendo el Ajuste de Prompts

El ajuste de prompts funciona anteponiendo "prompts suaves" entrenables a la entrada. Estos prompts suaves son vectores continuos que se optimizan durante el entrenamiento para ayudar al modelo a generar mejores salidas para tareas específicas. Este enfoque ofrece beneficios en términos de eficiencia: mantiene una huella de memoria mínima al almacenar solo los vectores de los prompts, preserva las capacidades generales del modelo y permite cambiar fácilmente entre tareas modificando los prompts en lugar de cargar copias completas del modelo.

## Implementación con PEFT

La biblioteca PEFT facilita la implementación del ajuste de prompts. Aquí tienes un ejemplo básico:

```python
from peft import PromptTuningConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Cargar el modelo base
model = AutoModelForCausalLM.from_pretrained("your-base-model")
tokenizer = AutoTokenizer.from_pretrained("your-base-model")

# Configurar el ajuste de prompts
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=8,  # Número de tokens entrenables
    prompt_tuning_init="TEXT",  # Inicialización desde texto
    prompt_tuning_init_text="Clasifica si este texto es positivo o negativo:",
    tokenizer_name_or_path="your-base-model",
)

# Crear el modelo ajustable con prompts
model = get_peft_model(model, peft_config)
```

## Componentes Clave

El núcleo del ajuste de prompts gira en torno a los tokens virtuales, que son incrustaciones entrenables agregadas a la entrada. Estos generalmente van de 8 a 32 tokens y pueden inicializarse de manera aleatoria o a partir de texto existente. El método de inicialización juega un papel crucial en el proceso de entrenamiento, siendo la inicialización basada en texto a menudo más efectiva que la inicialización aleatoria.

Durante el entrenamiento, solo los parámetros del prompt se actualizan, mientras que el modelo base permanece congelado. Este enfoque enfocado utiliza objetivos de entrenamiento estándar, pero requiere atención cuidadosa a la tasa de aprendizaje y al comportamiento de los gradientes de los tokens de los prompts.

## Comparación con Otros Métodos

Cuando se compara con otros enfoques de PEFT, el ajuste de prompts destaca por su eficiencia. Mientras que LoRA ofrece un bajo número de parámetros y uso de memoria pero requiere cargar adaptadores para cambiar de tarea, el ajuste de prompts logra un uso de recursos aún más bajo y permite un cambio inmediato de tareas. El ajuste completo, en cambio, demanda recursos significativos y requiere copias separadas del modelo para diferentes tareas.

| Método          | Parámetros  | Memoria | Cambio de Tarea |
|-----------------|-------------|---------|-----------------|
| Ajuste de Prompts | Muy Bajo   | Mínima  | Fácil           |
| LoRA            | Bajo        | Bajo    | Requiere Cargar |
| Ajuste Completo | Alto        | Alto    | Nueva Copia de Modelo |

Al implementar el ajuste de prompts, comienza con un número pequeño de tokens virtuales (8-16) y aumenta solo si la complejidad de la tarea lo exige. La inicialización basada en texto típicamente produce mejores resultados que la inicialización aleatoria, especialmente al usar texto relevante para la tarea. La estrategia de inicialización debe reflejar la complejidad de la tarea objetivo.

El entrenamiento requiere consideraciones ligeramente diferentes a las del ajuste completo. Las tasas de aprendizaje más altas suelen funcionar bien, pero es esencial monitorear cuidadosamente los gradientes de los tokens de los prompts. La validación regular con ejemplos diversos ayuda a garantizar un rendimiento robusto en diferentes escenarios.

## Aplicaciones

El ajuste de prompts destaca en varios escenarios, especialmente para tareas de clasificación y generación simple. Sus mínimos requisitos de recursos lo hacen ideal para entornos con recursos computacionales limitados. La capacidad de cambiar rápidamente entre tareas también lo hace valioso para aplicaciones multitarea donde se requieren comportamientos diferentes en momentos distintos.

## Próximos Pasos

Para obtener experiencia práctica con el ajuste de prompts, prueba el [Tutorial de Ajuste de Prompts](./notebooks/prompt_tuning_example.ipynb). Esta guía práctica te llevará paso a paso a implementar la técnica con tu propio modelo y datos.

## Recursos
- [Documentación de PEFT](https://huggingface.co/docs/peft)
- [Papel de Ajuste de Prompts](https://arxiv.org/abs/2104.08691)
- [Cookbook de Hugging Face](https://huggingface.co/learn/cookbook/prompt_tuning_peft)