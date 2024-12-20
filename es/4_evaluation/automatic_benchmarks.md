# Benchmarks Automáticos

Los "benchmarks" automáticos sirven como herramientas estandarizadas para evaluar modelos de lenguaje en diferentes tareas y capacidades. Aunque proporcionan un punto de partida útil para entender el rendimiento de un modelo, es importante reconocer que representan solo una parte de una estrategia de evaluación integral.

## Entendiendo los Benchmarks Automáticos

Los "benchmarks" automáticos generalmente consisten en conjuntos de datos curados con tareas y métricas de evaluación predefinidas. Estos "benchmarks" buscan evaluar diversos aspectos de la capacidad del modelo, desde la comprensión básica del lenguaje hasta el razonamiento complejo. La principal ventaja de usar "benchmarks" automáticos es su estandarización: permiten comparaciones consistentes entre diferentes modelos y proporcionan resultados reproducibles.

Sin embargo, es crucial entender que el rendimiento en un "benchmark" no siempre se traduce directamente en efectividad en el mundo real. Un modelo que sobresale en "benchmarks" académicos puede tener dificultades en aplicaciones específicas de dominio o casos de uso prácticos.

## Benchmarks y Sus Limitaciones

### Benchmarks de Conocimientos Generales

MMLU (Massive Multitask Language Understanding) evalúa conocimientos en 57 materias, desde ciencias hasta humanidades. Aunque es extenso, puede no reflejar la profundidad de experiencia necesaria para dominios específicos. TruthfulQA evalúa la tendencia de un modelo a reproducir conceptos erróneos comunes, aunque no puede capturar todas las formas de desinformación.

### Benchmarks de Razonamiento

BBH (Big Bench Hard) y GSM8K se enfocan en tareas de razonamiento complejo. BBH evalúa el pensamiento lógico y la planificación, mientras que GSM8K se centra específicamente en la resolución de problemas matemáticos. Estos "benchmarks" ayudan a evaluar capacidades analíticas pero pueden no capturar el razonamiento matizado requerido en escenarios del mundo real.

### Comprensión del Lenguaje

HELM proporciona un marco de evaluación holístico, mientras que WinoGrande prueba el sentido común mediante la desambiguación de pronombres. Estos "benchmarks" ofrecen información sobre las capacidades de procesamiento del lenguaje, pero pueden no representar completamente la complejidad de las conversaciones naturales o la terminología específica de un dominio.

## Enfoques Alternativos de Evaluación

Muchas organizaciones han desarrollado métodos de evaluación alternativos para abordar las limitaciones de los "benchmarks" estándar:

### Model de lenguaje como Juez (LLM-as-Judge)

Usar un modelo de lenguaje para evaluar los "outputs" de otro modelo se ha vuelto cada vez más popular. Este enfoque puede proporcionar retroalimentación más matizada que las métricas tradicionales, aunque viene con sus propios sesgos y limitaciones.

### Arenas de Evaluación (Evaluation Arenas)

Plataformas como Chatbot Arena permiten que los modelos interactúen y se evalúen entre sí en entornos controlados. Esto puede revelar fortalezas y debilidades que podrían no ser evidentes en "benchmarks" tradicionales.

### Grupos de Benchmarks Personalizados

Las organizaciones a menudo desarrollan conjuntos de "benchmarks" internos adaptados a sus necesidades específicas y casos de uso. Estos pueden incluir pruebas de conocimientos específicos de dominio o escenarios de evaluación que reflejen las condiciones reales de despliegue.

## Creando Tu Propia Estrategia de Evaluación

Recuerda que aunque LightEval facilita ejecutar "benchmarks" estándar, también deberías invertir tiempo en desarrollar métodos de evaluación específicos para tu caso de uso.

Aunque los "benchmarks" estándares proporcionan una línea base útil, no deberían ser tu único método de evaluación. Así es como puedes desarrollar un enfoque más integral:

1. Comienza con "benchmarks" estándares relevantes para establecer una línea base y permitir comparaciones con otros modelos.

2. Identifica los requisitos y desafíos específicos de tu caso de uso. ¿Qué tareas realizará tu modelo realmente? ¿Qué tipos de errores serían más problemáticos?

3. Desarrolla conjuntos de datos de evaluación personalizados que reflejen tu caso de uso real. Esto podría incluir:
   - Consultas reales de usuarios en tu dominio
   - Casos límite comunes que hayas encontrado
   - Ejemplos de escenarios particularmente desafiantes

4. Considera implementar una estrategia de evaluación por capas:
   - Métricas automatizadas para retroalimentación rápida
   - Evaluación humana para una comprensión más matizada
   - Revisión por expertos en el dominio para aplicaciones especializadas
   - Pruebas A/B en entornos controlados

## Usando LightEval para "Benchmarking"

Las tareas de LightEval se definen usando un formato específico:
``` 
{suite}|{task}|{num_few_shot}|{auto_reduce} 
```

- **suite**: El conjunto de "benchmarks" (por ejemplo, 'mmlu', 'truthfulqa')
- **task**: Tarea específica dentro del conjunto (por ejemplo, 'abstract_algebra')
- **num_few_shot**: Número de ejemplos a incluir en el "prompt" (0 para zero-shot)
- **auto_reduce**: Si se debe reducir automáticamente los ejemplos "few-shot" si el "prompt" es demasiado largo (0 o 1)

Ejemplo: `"mmlu|abstract_algebra|0|0"` evalúa la tarea de álgebra abstracta de MMLU sin ejempls en el "prompt" (i.e. "zero-shot").

### Ejemplo de Pipeline de Evaluación

Aquí tienes un ejemplo completo de configuración y ejecución de una evaluación utilizando "benchmarks" automáticos relevantes para un dominio específico:

```python
from lighteval.tasks import Task, Pipeline
from transformers import AutoModelForCausalLM

# Definir tareas para evaluar
domain_tasks = [
    "mmlu|anatomy|0|0",
    "mmlu|high_school_biology|0|0", 
    "mmlu|high_school_chemistry|0|0",
    "mmlu|professional_medicine|0|0"
]

# Configurar parámetros del pipeline
pipeline_params = {
    "max_samples": 40,  # Número de muestras para evaluar
    "batch_size": 1,    # Tamaño del "batch" para la inferencia
    "num_workers": 4    # Número de procesos paralelos
}

# Crear tracker de evaluación
evaluation_tracker = EvaluationTracker(
    output_path="./results",
    save_generations=True
)

# Cargar modelo y crear pipeline
model = AutoModelForCausalLM.from_pretrained("your-model-name")
pipeline = Pipeline(
    tasks=domain_tasks,
    pipeline_parameters=pipeline_params,
    evaluation_tracker=evaluation_tracker,
    model=model
)

# Ejecutar evaluación
pipeline.evaluate()

# Obtener y mostrar resultados
results = pipeline.get_results()
pipeline.show_results()
```

Los resultados se muestran en formato tabular:
``` 
|                  Task                  |Version|Metric|Value |   |Stderr|
|----------------------------------------|------:|------|-----:|---|-----:|
|all                                     |       |acc   |0.3333|±  |0.1169|
|leaderboard:mmlu:_average:5             |       |acc   |0.3400|±  |0.1121|
|leaderboard:mmlu:anatomy:5              |      0|acc   |0.4500|±  |0.1141|
|leaderboard:mmlu:high_school_biology:5  |      0|acc   |0.1500|±  |0.0819|
```

También puedes manejar los resultados en un DataFrame de pandas y visualizarlos o representarlos como prefieras.

# Próximos Pasos

⏩ Explora [Evaluación Personalizada en un Dominio](./custom_evaluation.md) para aprender a crear flujos de evaluación adaptados a tus necesidades específicas.
