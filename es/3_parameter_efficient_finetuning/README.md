# Alineación de Preferencias

Este módulo cubre técnicas para alinear los modelos de lenguaje con las preferencias humanas. Mientras que el ajuste supervisado ayuda a los modelos a aprender tareas, la alineación de preferencias fomenta que las salidas coincidan con las expectativas y valores humanos.

## Visión General

Los métodos típicos de alineación involucran varias etapas:
1. Ajuste Supervisado (SFT) para adaptar los modelos a dominios específicos.
2. Alineación de preferencias (como RLHF o DPO) para mejorar la calidad de las respuestas.

Enfoques alternativos como ORPO combinan el ajuste de instrucciones y la alineación de preferencias en un solo proceso. Aquí, nos enfocaremos en los algoritmos DPO y ORPO.

Si deseas aprender más sobre las diferentes técnicas de alineación, puedes leer más sobre ellas en el [Blog de Argilla](https://argilla.io/blog/mantisnlp-rlhf-part-8).

### 1️⃣ Optimización Directa de Preferencias (DPO)

La Optimización Directa de Preferencias (DPO) simplifica la alineación de preferencias optimizando directamente los modelos utilizando datos de preferencias. Este enfoque elimina la necesidad de modelos de recompensa separados y de un complejo aprendizaje por refuerzo, lo que lo hace más estable y eficiente que el aprendizaje por refuerzo de retroalimentación humana (RLHF) tradicional. Para más detalles, puedes consultar la [documentación de DPO](./dpo.md).

### 2️⃣ Optimización de Preferencias por Razón de Probabilidades (ORPO)

ORPO introduce un enfoque combinado para el ajuste de instrucciones y la alineación de preferencias en un solo proceso. Modifica el objetivo estándar de modelado de lenguaje al combinar la pérdida de log-verosimilitud negativa con un término de razón de probabilidades a nivel de tokens. El enfoque presenta un proceso de entrenamiento unificado de una sola etapa, una arquitectura sin referencia al modelo y una mayor eficiencia computacional. ORPO ha mostrado resultados impresionantes en varios puntos de referencia, demostrando un mejor rendimiento en AlpacaEval en comparación con los métodos tradicionales. Para más detalles, puedes consultar la [documentación de ORPO](./orpo.md).

## Notebook de Ejercicios

| Título           | Descripción | Ejercicio | Enlace | Colab |
|------------------|-------------|-----------|--------|-------|
| Entrenamiento DPO | Aprende cómo entrenar modelos usando Optimización Directa de Preferencias | 🐢 Entrenar un modelo usando el conjunto de datos Anthropic HH-RLHF<br>🐕 Usar tu propio conjunto de datos de preferencias<br>🦁 Experimentar con diferentes conjuntos de datos de preferencias y tamaños de modelo | [Notebook](./notebooks/dpo_finetuning_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/2_preference_alignment/notebooks/dpo_finetuning_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Abrir en Colab"/></a> |
| Entrenamiento ORPO | Aprende cómo entrenar modelos usando Optimización de Preferencias por Razón de Probabilidades | 🐢 Entrenar un modelo usando datos de instrucciones y preferencias<br>🐕 Experimentar con diferentes ponderaciones de pérdidas<br>🦁 Comparar los resultados de ORPO con DPO | [Notebook](./notebooks/orpo_finetuning_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/2_preference_alignment/notebooks/orpo_finetuning_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Abrir en Colab"/></a> |

## Recursos

- [Documentación de TRL](https://huggingface.co/docs/trl/index) - Documentación de la biblioteca Transformers Reinforcement Learning (TRL), que implementa varias técnicas de alineación, incluyendo DPO.
- [Papel de DPO](https://arxiv.org/abs/2305.18290) - Artículo de investigación original que introduce la Optimización Directa de Preferencias como una alternativa más simple al RLHF, optimizando directamente los modelos de lenguaje utilizando datos de preferencias.
- [Papel de ORPO](https://arxiv.org/abs/2402.01714) - Introduce la Optimización de Preferencias por Razón de Probabilidades, un enfoque novedoso que combina el ajuste de instrucciones y la alineación de preferencias en una sola etapa de entrenamiento.
- [Guía de RLHF de Argilla](https://argilla.io/blog/mantisnlp-rlhf-part-8/) - Una guía que explica diferentes técnicas de alineación, incluyendo RLHF, DPO y sus implementaciones prácticas.
- [Blog sobre DPO](https://huggingface.co/blog/dpo-trl) - Guía práctica sobre cómo implementar DPO utilizando la biblioteca TRL con ejemplos de código y mejores prácticas.
- [Script de ejemplo de TRL sobre DPO](https://github.com/huggingface/trl/blob/main/examples/scripts/dpo.py) - Script de ejemplo completo que muestra cómo implementar el entrenamiento DPO utilizando la biblioteca TRL.
- [Script de ejemplo de TRL sobre ORPO](https://github.com/huggingface/trl/blob/main/examples/scripts/orpo.py) - Implementación de referencia del entrenamiento ORPO utilizando la biblioteca TRL con opciones detalladas de configuración.
- [Manual de Alineación de Hugging Face](https://github.com/huggingface/alignment-handbook) - Guías de recursos y base de código para alinear modelos de lenguaje utilizando diversas técnicas, incluyendo SFT, DPO y RLHF.