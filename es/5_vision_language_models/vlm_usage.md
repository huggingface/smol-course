# Modelos de Visión-Lenguaje

Los Modelos de Visión-Lenguaje (VLMs) cierran la brecha entre imágenes y texto, permitiendo tareas avanzadas como generar descripciones de imágenes, responder preguntas basadas en elementos visuales o comprender la relación entre datos textuales y visuales. Su arquitectura está diseñada para procesar ambas modalidades de manera integrada.

### Arquitectura

Los VLMs combinan componentes de procesamiento de imágenes con modelos de generación de texto para lograr una comprensión unificada. Los elementos principales de su arquitectura son:

![Arquitectura de VLM](./images/VLM_Architecture.png)

- **Codificador de Imágenes**: Transforma imágenes en representaciones numéricas compactas. Se utilizan comúnmente codificadores preentrenados como CLIP o vision transformers (ViT).
- **Proyector de Embeddings**: Mapea características de imagen a un espacio compatible con embeddings textuales, utilizando capas densas o transformaciones lineales.
- **Decodificador de Texto**: Actúa como el componente de generación de lenguaje, traduciendo información multimodal en texto coherente. Ejemplos incluyen modelos generativos como Llama o Vicuna.
- **Proyector Multimodal**: Proporciona una capa adicional para combinar representaciones de imagen y texto. Es fundamental para modelos como LLaVA en la creación de conexiones más sólidas entre ambas modalidades.

La mayoría de los VLMs aprovechan codificadores de imagen y decodificadores de texto preentrenados, alineándolos mediante fine-tuning en conjuntos de datos de pares imagen-texto. Este enfoque optimiza el entrenamiento y permite que los modelos generalicen de manera efectiva.

### Uso

![Proceso de VLM](./images/VLM_Process.png)

Los VLMs se aplican a una variedad de tareas multimodales. Su adaptabilidad les permite desempeñarse en distintos campos con diferentes niveles de fine-tuning:

- **Generación de Descripciones de Imágenes**: Creación de textos descriptivos a partir de imágenes.
- **Respuesta a Preguntas Visuales (VQA por sus siglas en inglés)**: Responder preguntas sobre el contenido de una imagen.
- **Recuperación Cruzada de Modalidades**: Encontrar texto correspondiente a una imagen o viceversa.
- **Aplicaciones Creativas**: Asistencia en diseño, generación artística o creación de contenido multimedia interactivo.

![Uso de VLM](./images/VLM_Usage.png)

El entrenamiento y fine-tuning de VLMs depende de la disponibilidad conjuntos de datos de alta calidad que vinculen imágenes con anotaciones textuales. Herramientas como la librería `transformers` de Hugging Face facilitan el acceso a VLMs preentrenados y proporcionan flujos de trabajo optimizados para fine-tuning personalizado.

### Formato de Chat

Muchos VLMs están estructurados para interactuar de manera conversacional, mejorando la experiencia del usuario. Este formato incluye:

- Un **mensaje del sistema** que define el rol o contexto del modelo, como "Eres un asistente que analiza datos visuales".
- **Consultas del usuario** que combinan entradas textuales con imágenes asociadas.
- **Respuestas del asistente** que proporcionan salidas textuales derivadas del análisis multimodal.

Este esquema conversacional es intuitivo y se alinea con las expectativas de los usuarios, especialmente en aplicaciones interactivas como servicio al cliente o herramientas educativas.

Aquí tienes un ejemplo de cómo se estructura una entrada formateada:

```json
[
    {
        "role": "system",
        "content": [{"type": "text", "text": "Eres un Modelo de Visión-Lenguaje especializado en interpretar datos visuales de gráficos..."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "<image_data>"},
            {"type": "text", "text": "¿Cuál es el valor más alto en el gráfico de barras?"}
        ]
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "42"}]
    }
]
```

**Trabajo con Múltiples Imágenes y Videos**

Los VLMs también pueden procesar múltiples imágenes o incluso videos adaptando la estructura de entrada para manejar entradas visuales secuenciales o en paralelo. Para videos, se pueden extraer cuadros y procesarlos como imágenes individuales, manteniendo el orden temporal.

## Recursos

- [Hugging Face Blog: Modelos de Visión-Lenguaje](https://huggingface.co/blog/vlms)
- [Hugging Face Blog: SmolVLM](https://huggingface.co/blog/smolvlm)

## Próximos Pasos

⏩ Prueba el [vlm_usage_sample.ipynb](./notebooks/vlm_usage_sample.ipynb) para experimentar con diferentes usos de SMOLVLM.
