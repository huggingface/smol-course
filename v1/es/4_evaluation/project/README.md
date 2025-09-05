# Evaluación Específica en un Dominio con Argilla, Distilabel y LightEval

La mayoría de los "benchmarks" populares evalúan capacidades muy generales (razonamiento, matemáticas, código), pero ¿alguna vez has necesitado estudiar capacidades más específicas?

¿Qué deberías hacer si necesitas evaluar un modelo en un **dominio específico** relevante para tus casos de uso? (Por ejemplo, aplicaciones financieras, legales o médicas).

Este tutorial muestra todo el flujo de trabajo que puedes seguir, desde la creación de datos relevantes y la anotación de tus muestras hasta la evaluación de tu modelo, utilizando las herramientas de [Argilla](https://github.com/argilla-io/argilla), [distilabel](https://github.com/argilla-io/distilabel) y [lighteval](https://github.com/huggingface/lighteval). Para nuestro ejemplo, nos centraremos en generar preguntas de exámenes a partir de múltiples documentos.

## Estructura del Proyecto

Seguiremos 4 pasos, con un script para cada uno: generar un conjunto de datos, anotarlo, extraer muestras relevantes para la evaluación y, finalmente, evaluar los modelos.

| Nombre del Script       | Descripción                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| generate_dataset.py     | Genera preguntas de exámenes a partir de múltiples documentos de texto usando un modelo de lenguaje especificado. |
| annotate_dataset.py     | Crea un conjunto de datos en Argilla para la anotación manual de las preguntas generadas. |
| create_dataset.py       | Procesa los datos anotados desde Argilla y crea una base de datos de Hugging Face. |
| evaluation_task.py      | Define una tarea personalizada en LightEval para evaluar modelos de lenguaje en el conjunto de preguntas de examen. |

## Pasos

### 1. Generar el Conjunto de Datos

El script `generate_dataset.py` utiliza la biblioteca distilabel para generar preguntas de examen basadas en múltiples documentos de texto. Usa el modelo especificado (por defecto: Meta-Llama-3.1-8B-Instruct) para crear preguntas, respuestas correctas y respuestas incorrectas (llamadas "distractores"). Puedes agregar tus propios datos y también usar un modelo diferente.

Para ejecutar la generación:

```sh
python generate_dataset.py --input_dir ubicacion/de/los/documentos --model_id id_del_modelo --output_path output_directory
```

Esto creará un [Distiset](https://distilabel.argilla.io/dev/sections/how_to_guides/advanced/distiset/) que contiene las preguntas de examen generadas para todos los documentos en el directorio de entrada.

### 2. Anotar el Conjunto de Datos

El script `annotate_dataset.py` toma las preguntas generadas y crea una base de datos en Argilla para su anotación. Este script configura la estructura de la base de datos y la completa con las preguntas y respuestas generadas, aleatorizando el orden de las respuestas para evitar sesgos. Una vez en Argilla, tú o un experto en el dominio pueden validar el conjunto de datos con las respuestas correctas.

Verás respuestas sugeridas por el LLM en orden aleatorio, y podrás aprobar la respuesta correcta o seleccionar una diferente. La duración de este proceso dependerá de la escala de tu base de datos de evaluación, la complejidad de tus datos y la calidad de tu LLM. Por ejemplo, logramos crear 150 muestras en 1 hora en el dominio de aprendizaje por transferencia, utilizando Llama-3.1-70B-Instruct, principalmente aprobando respuestas correctas y descartando las incorrectas.

Para ejecutar el proceso de anotación:

```sh
python annotate_dataset.py --dataset_path ubicacion/del/distiset --output_dataset_name nombre_de_los_datos_argilla
```

Esto creará un conjunto de datos en Argilla que puede ser utilizado para revisión y anotación manual.

![argilla_dataset](./images/domain_eval_argilla_view.png)

Si no estás usando Argilla, desplégalo localmente o en Spaces siguiendo esta [guía rápida](https://docs.argilla.io/latest/getting_started/quickstart/).

### 3. Crear la Base de Datos

El script `create_dataset.py` procesa los datos anotados desde Argilla y crea una base de datos en Hugging Face. El script incorpora tanto respuestas sugeridas como respuestas anotadas manualmente. El script creará una base de datos con la pregunta, posibles respuestas y el nombre de la columna para la respuesta correcta. Para crear la base de datos final:

```sh
huggingface_hub login
python create_dataset.py --dataset_path nombre_de_los_datos_argilla --dataset_repo_id id_repo_hf
```

Esto enviará el conjunto de datos al Hugging Face Hub bajo el repositorio especificado. Puedes ver una base de datos de ejemplo en el hub [aquí](https://huggingface.co/datasets/burtenshaw/exam_questions/viewer/default/train). La vista previa de la base de datos se ve así:

![hf_dataset](./images/domain_eval_dataset_viewer.png)

### 4. Tarea de Evaluación

El script `evaluation_task.py` define una tarea personalizada en LightEval para evaluar modelos de lenguaje en el conjunto de preguntas de examen. Incluye una función de "prompt", una métrica de "accuracy" personalizada y la configuración de la tarea.

Para evaluar un modelo utilizando lighteval con la tarea personalizada de preguntas de examen:

```sh
lighteval accelerate \
    --model_args "pretrained=HuggingFaceH4/zephyr-7b-beta" \
    --tasks "community|exam_questions|0|0" \
    --custom_tasks domain-eval/evaluation_task.py \
    --output_dir "./evals"
```

Puedes encontrar guías detalladas en el wiki de lighteval sobre cada uno de estos pasos:

- [Crear una Tarea Personalizada](https://github.com/huggingface/lighteval/wiki/Adding-a-Custom-Task)
- [Crear una Métrica Personalizada](https://github.com/huggingface/lighteval/wiki/Adding-a-New-Metric)
- [Usar Métricas Existentes](https://github.com/huggingface/lighteval/wiki/Metric-List)
