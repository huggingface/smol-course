# Alineación de Preferencias

Este módulo cubre técnicas para alinear modelos de lenguaje con las preferencias humanas. Mientras que la afinación supervisada (SFT) ayuda a los modelos a aprender tareas, la alineación de preferencias fomenta que las salidas coincidan con las expectativas y valores humanos.

## Descripción General

Los métodos típicos de alineación incluyen múltiples etapas:
1. Afinación Supervisada (SFT) para adaptar los modelos a dominios específicos.
2. Alineación de preferencias (como RLHF o DPO) para mejorar la calidad de las respuestas.

Enfoques alternativos como ORPO combinan la afinación por instrucciones y la alineación de preferencias en un solo proceso. Aquí, nos enfocaremos en los algoritmos DPO y ORPO.

Si deseas aprender más sobre las diferentes técnicas de alineación, puedes leer más sobre ellas en el [Blog de Argilla](https://argilla.io/blog/mantisnlp-rlhf-part-8).

### 1️⃣ Optimización Directa de Preferencias (DPO)

La Optimización Directa de Preferencias (DPO) simplifica la alineación de preferencias optimizando directamente los modelos utilizando datos de preferencias. Este enfoque elimina la necesidad de modelos de recompensa separados y de un aprendizaje por refuerzo complejo, lo que lo hace más estable y eficiente que el Aprendizaje por Refuerzo de Retroalimentación Humana (RLHF) tradicional. Para más detalles, puedes consultar la [documentación de Optimización Directa de Preferencias (DPO)](./dpo.md).

### 2️⃣ Optimización de Preferencias por Ratio de Probabilidades (ORPO)

ORPO introduce un enfoque combinado para la afinación por instrucciones y la alineación de preferencias en un solo proceso. Modifica el objetivo estándar del modelado de lenguaje combinando la pérdida de verosimilitud logarítmica negativa con un término de ratio de probabilidades a nivel de token. El enfoque presenta un proceso de entrenamiento de una sola etapa, una arquitectura libre de modelo de referencia y una mayor eficiencia computacional. ORPO ha mostrado resultados impresionantes en varios puntos de referencia, demostrando un mejor rendimiento en AlpacaEval en comparación con los métodos tradicionales. Para más detalles, puedes consultar la [documentación de Optimización de Preferencias por Ratio de Probabilidades (ORPO)](./orpo.md).

## Notebooks de Ejercicios

| Titulo | Descripción | Ejercicio | Enlace | Colab |
|-------|-------------|----------|------|-------|
| Entrenamiento DPO | Aprende a entrenar modelos utilizando la Optimización Directa de Preferencias | 🐢 Entrenar un modelo utilizando el conjunto de datos HH-RLHF de Anthropic<br>🐕 Utiliza tu propio conjunto de datos de preferencias<br>🦁 Experimenta con diferentes conjuntos de datos de preferencias y tamaños de modelo | [Notebook](./notebooks/dpo_finetuning_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/2_preference_alignment/notebooks/dpo_finetuning_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Abrir en Colab"/></a> |
| Entrenamiento ORPO | Aprende a entrenar modelos utilizando la Optimización de Preferencias por Ratio de Probabilidades | 🐢 Entrenar un modelo utilizando datos de instrucciones y preferencias<br>🐕 Experimenta con diferentes ponderaciones de la pérdida<br>🦁 Compara los resultados de ORPO con DPO | [Notebook](./notebooks/orpo_finetuning_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/2_preference_alignment/notebooks/orpo_finetuning_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Abrir en Colab"/></a> |

## Recursos

- [Documentación de TRL](https://huggingface.co/docs/trl/index) - Documentación para la librería Transformers Reinforcement Learning (TRL), que implementa diversas técnicas de alineación, incluyendo DPO.
- [Papel de DPO](https://arxiv.org/abs/2305.18290) - Artículo original de investigación que introduce la Optimización Directa de Preferencias como una alternativa más simple al RLHF que optimiza directamente los modelos de lenguaje utilizando datos de preferencias.
- [Papel de ORPO](https://arxiv.org/abs/2403.07691) - Introduce la Optimización de Preferencias por Ratio de Probabilidades, un enfoque novedoso que combina la afinación por instrucciones y la alineación de preferencias en una sola etapa de entrenamiento.
- [Guía de RLHF de Argilla](https://argilla.io/blog/mantisnlp-rlhf-part-8/) - Una guía que explica diferentes técnicas de alineación, incluyendo RLHF, DPO y sus implementaciones prácticas.
- [Entrada en el Blog sobre DPO](https://huggingface.co/blog/dpo-trl) - Guía práctica sobre cómo implementar DPO utilizando la librería TRL con ejemplos de código y mejores prácticas.
- [Script de ejemplo de TRL sobre DPO](https://github.com/huggingface/trl/blob/main/examples/scripts/dpo.py) - Script completo de ejemplo que demuestra cómo implementar el entrenamiento DPO utilizando la librería TRL.
- [Script de ejemplo de TRL sobre ORPO](https://github.com/huggingface/trl/blob/main/examples/scripts/orpo.py) - Implementación de referencia del entrenamiento ORPO utilizando la librería TRL con opciones detalladas de configuración.
- [Manual de Alineación de Hugging Face](https://github.com/huggingface/alignment-handbook) - Guías y código para alinear modelos de lenguaje utilizando diversas técnicas, incluyendo SFT, DPO y RLHF.
