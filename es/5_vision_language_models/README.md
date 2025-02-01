# Modelos de Lenguaje Visual (VLM)

## 1. Uso de VLM

Los Modelos de Lenguaje Visuales (VLMs, por sus siglas en inglés) procesan entradas de imágenes junto con texto para habilitar tareas como la descripción de imágenes, respuestas a preguntas visuales y razonamiento multimodal.

Una arquitectura típica de VLM consta de un codificador de imágenes para extraer características visuales, una capa de proyección para alinear representaciones visuales y textuales, y un modelo de lenguaje para procesar o generar texto. Esto permite que el modelo establezca conexiones entre elementos visuales y conceptos lingüísticos.

Los VLMs pueden usarse en diferentes configuraciones dependiendo del caso de uso. Los modelos base manejan tareas generales de visión y lenguaje, mientras que las variantes optimizadas para chat soportan interacciones conversacionales. Algunos modelos incluyen componentes adicionales para anclar las predicciones en evidencia visual o especializarse en tareas específicas como la detección de objetos.

Para más detalles sobre la parte técnica y el uso de VLMs, consulta la página de [Uso de VLM](./vlm_usage.md).

## 2. Fine-Tuning de VLM

El fine-tuning de un VLM implica adaptar un modelo preentrenado para realizar tareas específicas o para operar eficazmente en un conjunto de datos particular. El proceso puede seguir metodologías como el fine-tuning supervisado, optimización por preferencias o un enfoque híbrido que combine ambos, como se introdujo en los Módulos 1 y 2.

Aunque las herramientas y técnicas fundamentales son similares a las utilizadas para los LLMs, el fine-tuning de VLMs requiere un enfoque adicional en la representación y preparación de datos para imágenes. Esto asegura que el modelo integre y procese de manera efectiva tanto los datos visuales como textuales para un rendimiento óptimo. Dado que el modelo de demostración, SmolVLM, es significativamente más grande que el modelo de lenguaje utilizado en el módulo anterior, es esencial explorar métodos para un fine-tuning eficiente. Técnicas como la cuantización y PEFT pueden ayudar a hacer el proceso más accesible y rentable, permitiendo que más usuarios experimenten con el modelo.

Para obtener una guía detallada sobre el fine-tuning de VLMs, visita la página de [Fine-Tuning de VLM](./vlm_finetuning.md).

## Cuadernos de Ejercicios

| Título | Descripción | Ejercicio | Enlace | Colab |
|--------|-------------|-----------|--------|-------|
| Uso de VLM | Aprende cómo cargar y usar un VLM preentrenado para diversas tareas | 🐢 Procesar una imagen<br>🐕 Procesar múltiples imágenes con manejo de lotes <br>🦁 Procesar un video completo | [Cuaderno](./notebooks/vlm_usage_sample.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/5_vision_language_models/notebooks/vlm_usage_sample.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Abrir en Colab"/></a> |
| Fine-Tuning de VLM | Aprende cómo realizar el fine-tuning de un VLM preentrenado para conjuntos de datos específicos | 🐢 Usar un conjunto de datos básico para el fine-tuning<br>🐕 Probar un nuevo conjunto de datos<br>🦁 Experimentar con métodos alternativos de fine-tuning | [Cuaderno](./notebooks/vlm_sft_sample.ipynb)| <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/5_vision_language_models/notebooks/vlm_sft_sample.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Abrir en Colab"/></a> |

## Referencias  
- [Hugging Face Learn: Fine-Tuning Supervisado de VLMs](https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl)
- [Hugging Face Learn: Fine-Tuning Supervisado de SmolVLM](https://huggingface.co/learn/cookbook/fine_tuning_smol_vlm_sft_trl)  
- [Hugging Face Learn: Fine-Tuning por Optimización de Preferencias SmolVLM](https://huggingface.co/learn/cookbook/fine_tuning_vlm_dpo_smolvlm_instruct)  
- [Hugging Face Blog: Optimización de Preferencias para VLMs](https://huggingface.co/blog/dpo_vlm)
- [Hugging Face Blog: Modelos de Lenguaje Visuales](https://huggingface.co/blog/vlms)
- [Hugging Face Blog: SmolVLM](https://huggingface.co/blog/smolvlm)  
- [Hugging Face Model: SmolVLM-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)
- [CLIP: Aprendiendo Modelos Visual Transferibles desde Supervisión en Lenguaje Natural](https://arxiv.org/abs/2103.00020)  
- [Align Before Fuse: Aprendizaje de Representaciones de Visión y Lenguaje con Destilación por Momentum](https://arxiv.org/abs/2107.07651)
