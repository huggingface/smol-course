# Modelos de Visión-Lenguaje

## 1. Uso de VLM

Los Modelos de Visión-Lenguaje (VLMs por sus siglas en inglés) procesan imágenes y texto de manera simultánea para realizar tareas como la generación de descripciones de imágenes, la respuesta a preguntas visuales y el razonamiento multimodal.

Una arquitectura típica de VLM incluye un codificador de imágenes para extraer características visuales, una capa de proyección para alinear las representaciones visuales y textuales, y un modelo de lenguaje para procesar o generar texto. Esto permite al modelo establecer conexiones entre elementos visuales y conceptos lingüísticos.

Los VLMs pueden configurarse según el caso de uso. Los modelos base manejan tareas generales de visión-lenguaje, mientras que las variantes optimizadas para chat permiten interacciones conversacionales. Algunos modelos incluyen componentes adicionales para fundamentar predicciones en evidencia visual o especializarse en tareas específicas, como la detección de objetos.

Para más detalles sobre el uso y las técnicas de VLMs, consulta la página [Uso de VLM](./vlm_usage.md).

## 2. Fine-tuning de VLM

El fine-tuning de un VLM consiste en adaptar un modelo previamente entrenado para realizar tareas específicas o mejorar su desempeño en un conjunto de datos determinado. Este proceso puede seguir metodologías como el fine-tuning supervisado, la optimización por preferencias o un enfoque híbrido que combine ambos, como se introduce en los Módulos 1 y 2.

Aunque las herramientas y técnicas principales son similares a las utilizadas en los LLMs, el fine-tuning de VLMs requiere un enfoque adicional en la representación y preparación de datos de imágenes. Esto garantiza que el modelo integre y procese de manera efectiva tanto los datos visuales como los textuales para un rendimiento óptimo. Dado que el modelo de demostración, SmolVLM, es significativamente más grande que el modelo de lenguaje utilizado en el módulo anterior, es fundamental explorar métodos para un fine-tuning eficiente. Técnicas como la cuantización y PEFT pueden hacer que el proceso sea más accesible y rentable, permitiendo que más usuarios experimenten con el modelo.

Para obtener una guía detallada sobre el fine-tuning de VLMs, visita la página [fine-tuning de VLM](./vlm_finetuning.md).

## Cuadernos de ejercicios

| Título | Descripción | Ejercicio | Enlace | Colab |
|--------|-------------|-----------|--------|-------|
| Uso de VLM | Aprende a cargar y usar un VLM preentrenado para diversas tareas | 🐢 Procesar una imagen<br>🐕 Procesar múltiples imágenes con manejo por lotes<br>🦁 Procesar un video completo | [Notebook](./notebooks/vlm_usage_sample.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/5_vision_language_models/notebooks/vlm_usage_sample.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Abrir en Colab"/></a> |
| fine-tuning de VLM | Aprende a ajustar un VLM preentrenado para conjuntos de datos específicos | 🐢 Usar un conjunto de datos básico para fine-tuning<br>🐕 Probar un nuevo conjunto de datos<br>🦁 Experimentar con métodos alternativos de fine-tuning | [Notebook](./notebooks/vlm_sft_sample.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/5_vision_language_models/notebooks/vlm_sft_sample.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Abrir en Colab"/></a> |

## Referencias

- [Hugging Face Learn: fine-tuning supervisado de VLMs](https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl)
- [Hugging Face Learn: fine-tuning supervisado de SmolVLM](https://huggingface.co/learn/cookbook/fine_tuning_smol_vlm_sft_trl)
- [Hugging Face Learn: Optimización de preferencias para el fine-tuning de SmolVLM](https://huggingface.co/learn/cookbook/fine_tuning_vlm_dpo_smolvlm_instruct)
- [Hugging Face Blog: Optimización de preferencias para VLMs](https://huggingface.co/blog/dpo_vlm)
- [Hugging Face Blog: Modelos de Visión y Lenguaje](https://huggingface.co/blog/vlms)
- [Hugging Face Blog: SmolVLM](https://huggingface.co/blog/smolvlm)
- [Hugging Face Model: SmolVLM-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)
- [CLIP: Learning Transferable Visual Models from Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [Align Before Fuse: Vision and Language Representation Learning with Momentum Distillation](https://arxiv.org/abs/2107.07651)
