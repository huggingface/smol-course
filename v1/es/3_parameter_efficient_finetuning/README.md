# Fine-Tuning Eficiente en Parámetros (PEFT)

A medida que los modelos de lenguaje se hacen más grandes, el *fine-tuning* tradicional se vuelve cada vez más complicado. Afinar completamente un modelo con 1.7 mil millones de parámetros, por ejemplo, requiere una memoria de GPU significativa, hace costoso almacenar copias separadas del modelo y puede ocasionar un olvido catastrófico de las capacidades originales del modelo. Los métodos de *fine-tuning* eficiente en parámetros (*Parameter-Efficient Fine-Tuning* o PEFT) abordan estos problemas modificando solo un subconjunto pequeño de los parámetros del modelo, mientras que la mayor parte del modelo permanece congelada.

El *fine-tuning* tradicional actualiza todos los parámetros del modelo durante el entrenamiento, lo cual resulta poco práctico para modelos grandes. Los métodos PEFT introducen enfoques para adaptar modelos utilizando una fracción mínima de parámetros entrenables, generalmente menos del 1% del tamaño original del modelo. Esta reducción dramática permite:

- Realizar *fine-tuning* en hardware de consumo con memoria de GPU limitada.
- Almacenar eficientemente múltiples adaptaciones de tareas específicas.
- Mejorar la generalización en escenarios con pocos datos.
- Entrenamientos y ciclos de iteración más rápidos.

## Métodos Disponibles

En este módulo, se cubrirán dos métodos populares de PEFT:

### 1️⃣ LoRA (Adaptación de Bajo Rango)

LoRA se ha convertido en el método PEFT más adoptado, ofreciendo una solución sofisticada para la adaptación eficiente de modelos. En lugar de modificar el modelo completo, **LoRA inyecta matrices entrenables en las capas de atención del modelo.** Este enfoque, por lo general, reduce los parámetros entrenables en aproximadamente un 90%, manteniendo un rendimiento comparable al *fine-tuning* completo. Exploraremos LoRA en la sección [LoRA (Adaptación de Bajo Rango)](./lora_adapters.md).

### 2️⃣ *Prompt Tuning*

El *prompt tuning* ofrece un enfoque **aún más ligero** al **añadir tokens entrenables a la entrada** en lugar de modificar los pesos del modelo. Aunque es menos popular que LoRA, puede ser útil para adaptar rápidamente un modelo a nuevas tareas o dominios. Exploraremos el *prompt tuning* en la sección [Prompt Tuning](./prompt_tuning.md).

## Notebooks de Ejercicios

| Título | Descripción | Ejercicio | Enlace | Colab |
|-------|-------------|-----------|--------|-------|
| *Fine-tuning* con LoRA | Aprende a realizar *fine-tuning* con adaptadores LoRA | 🐢 Entrena un modelo con LoRA<br>🐕 Experimenta con diferentes valores de rango<br>🦁 Compara el rendimiento con el *fine-tuning* completo | [Notebook](./notebooks/finetune_sft_peft.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/3_parameter_efficient_finetuning/notebooks/finetune_sft_peft.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Abrir en Colab"/></a> |
| Carga Adaptadores LoRA | Aprende a cargar y usar adaptadores LoRA entrenados | 🐢 Carga adaptadores preentrenados<br>🐕 Combina adaptadores con el modelo base<br>🦁 Alterna entre múltiples adaptadores | [Notebook](./notebooks/load_lora_adapter_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/3_parameter_efficient_finetuning/notebooks/load_lora_adapter_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Abrir en Colab"/></a> |

<!-- | Prompt Tuning | Aprende a implementar *prompt tuning* | 🐢 Entrenar *soft prompts*<br>🐕 Comparar diferentes estrategias de inicialización<br>🦁 Evaluar en múltiples tareas | [Notebook](./notebooks/prompt_tuning_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/3_parameter_efficient_finetuning/notebooks/prompt_tuning_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Abrir en Colab"/></a> | -->

## Recursos
- [Documentación de PEFT](https://huggingface.co/docs/peft)
- [Artículo de LoRA](https://arxiv.org/abs/2106.09685)
- [Artículo de QLoRA](https://arxiv.org/abs/2305.14314)
- [Artículo de *Prompt Tuning*](https://arxiv.org/abs/2104.08691)
- [Guía de PEFT en Hugging Face](https://huggingface.co/blog/peft)
- [Cómo hacer *Fine-Tuning* de LLMs en 2024 con Hugging Face](https://www.philschmid.de/fine-tune-llms-in-2024-with-trl)
- [TRL](https://huggingface.co/docs/trl/index)