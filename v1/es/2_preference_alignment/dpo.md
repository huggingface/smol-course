**Optimización Directa de Preferencias (DPO)**

La Optimización Directa de Preferencias (DPO) ofrece un enfoque simplificado para alinear modelos de lenguaje con las preferencias humanas. A diferencia de los métodos tradicionales de RLHF (Reinforcement Learning with Human Feedback), que requieren modelos de recompensas separados y algoritmos complejos de aprendizaje por refuerzo, DPO optimiza directamente el modelo utilizando datos de preferencia.

## Entendiendo DPO

DPO redefine la alineación de preferencias como un problema de clasificación basado en datos de preferencias humanas. Los enfoques tradicionales de RLHF requieren entrenar un modelo de recompensa separado y usar algoritmos complejos como PPO (Proximal Policy Optimization) para alinear las salidas del modelo. DPO simplifica este proceso al definir una función de pérdida que optimiza directamente la política del modelo en función de las salidas preferidas versus las no preferidas.

Este enfoque ha demostrado ser altamente efectivo en la práctica, utilizándose para entrenar modelos como Llama. Al eliminar la necesidad de un modelo de recompensa separado y una etapa de aprendizaje por refuerzo, DPO hace que la alineación de preferencias sea más accesible y estable.

## Cómo Funciona DPO

El proceso de DPO requiere un fine-tuning supervisado (SFT) para adaptar el modelo al dominio objetivo. Esto crea una base para el aprendizaje de preferencias mediante el entrenamiento en conjuntos de datos estándar de seguimiento de instrucciones. El modelo aprende a completar tareas básicas mientras mantiene sus capacidades generales.

Luego viene el aprendizaje de preferencias, donde el modelo se entrena con pares de salidas: una preferida y una no preferida. Los pares de preferencias ayudan al modelo a entender qué respuestas se alinean mejor con los valores y expectativas humanas.

La innovación central de DPO radica en su enfoque de optimización directa. En lugar de entrenar un modelo de recompensa separado, DPO utiliza una pérdida de entropía cruzada binaria para actualizar directamente los pesos del modelo en función de los datos de preferencia. Este proceso simplificado hace que el entrenamiento sea más estable y eficiente, logrando resultados comparables o mejores que los métodos tradicionales de RLHF.

## Conjuntos de Datos para DPO

Los conjuntos de datos para DPO se crean típicamente anotando pares de respuestas como preferidas o no preferidas. Esto se puede hacer manualmente o mediante técnicas de filtrado automatizado. A continuación, se muestra un ejemplo de la estructura de un conjunto de datos de preferencias para DPO en un turno único:

| Prompt | Elegido | Rechazado |
|--------|---------|-----------|
| ...    | ...     | ...       |
| ...    | ...     | ...       |
| ...    | ...     | ...       |

La columna `Prompt` contiene el prompt utilizado para generar las respuestas `Elegido` y `Rechazado`. Las columnas `Elegido` y `Rechazado` contienen las respuestas preferidas y no preferidas, respectivamente. Existen variaciones en esta estructura, por ejemplo, incluyendo una columna de `system_prompt` o `Input` con material de referencia. Los valores de `elegido` y `rechazado` pueden representarse como cadenas para conversaciones de un solo turno o como listas de conversación.

Puedes encontrar una colección de conjuntos de datos DPO en Hugging Face [aquí](https://huggingface.co/collections/argilla/preference-datasets-for-dpo-656f0ce6a00ad2dc33069478).

## Implementación con TRL

La biblioteca Transformers Reinforcement Learning (TRL) facilita la implementación de DPO. Las clases `DPOConfig` y `DPOTrainer` siguen la misma API de estilo `transformers`.

Aquí tienes un ejemplo básico de cómo configurar el entrenamiento de DPO:

```python
from trl import DPOConfig, DPOTrainer

# Definir los argumentos
training_args = DPOConfig(
    ...
)

# Inicializar el entrenador
trainer = DPOTrainer(
    model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    ...
)

# Entrenar el modelo
trainer.train()
```

En el [tutorial de DPO](./notebooks/dpo_finetuning_example.ipynb) se cubrirán más detalles sobre cómo utilizar las clases `DPOConfig` y `DPOTrainer`.

## Mejores Prácticas

La calidad de los datos es crucial para una implementación exitosa de DPO. El conjunto de datos de preferencias debe incluir ejemplos diversos que cubran diferentes aspectos del comportamiento deseado. Las pautas claras de anotación aseguran un etiquetado consistente de respuestas preferidas y no preferidas. Puedes mejorar el rendimiento del modelo mejorando la calidad de tu conjunto de datos de preferencias. Por ejemplo, filtrando conjuntos de datos más grandes para incluir solo ejemplos de alta calidad o que estén más relacionados con tu caso de uso.

Durante el entrenamiento, monitorea cuidadosamente la convergencia de la pérdida y valida el rendimiento con datos retenidos. Es posible que sea necesario ajustar el parámetro beta para equilibrar el aprendizaje de preferencias con el mantenimiento de las capacidades generales del modelo. La evaluación regular sobre ejemplos diversos ayuda a garantizar que el modelo está aprendiendo las preferencias deseadas sin sobreajustarse.

Compara las salidas del modelo con el modelo de referencia para verificar la mejora en la alineación de preferencias. Probar el modelo con una variedad de prompts, incluidos los casos extremos, ayuda a garantizar un aprendizaje robusto de preferencias en diferentes escenarios.

## Próximos Pasos

⏩ Para obtener experiencia práctica con DPO, prueba el [Tutorial de DPO](./notebooks/dpo_finetuning_example.ipynb). Esta guía práctica te guiará a través de la implementación de la alineación de preferencias con tu propio modelo, desde la preparación de datos hasta el entrenamiento y la evaluación.

⏭️ Después de completar el tutorial, puedes explorar la página de [ORPO](./orpo.md) para aprender sobre otra técnica de alineación de preferencias.