# Evaluación Personalizada en un Dominio

Aunque los "benchmarks" estándares proporcionan conocimiento relevante, muchas aplicaciones requieren enfoques de evaluación especializados que se adapten a dominios específicos o a casos de uso particulares. Esta guía te ayudará a crear flujos de evaluación personalizados que evalúen con precisión el rendimiento de tu modelo en tu dominio objetivo.

## Diseñando Tu Estrategia de Evaluación

Una estrategia de evaluación personalizada exitosa comienza con objetivos claros. Es fundamental considerar qué capacidades específicas necesita demostrar tu modelo en tu dominio. Esto podría incluir conocimientos técnicos, patrones de razonamiento o formatos específicos del dominio. Documenta estos requisitos cuidadosamente; ellos guiarán tanto el diseño de tus tareas como la selección de métricas.

Tu evaluación debe probar tanto casos de uso estándar como casos límite. Por ejemplo, en un dominio médico, podrías evaluar tanto escenarios comunes de diagnóstico como condiciones raras. En aplicaciones financieras, podrías probar tanto transacciones rutinarias como casos complejos que involucren múltiples monedas o condiciones especiales.

## Implementación con LightEval

LightEval proporciona un marco flexible para implementar evaluaciones personalizadas. Así es como puedes crear una tarea personalizada:

```python
from lighteval.tasks import Task, Doc
from lighteval.metrics import SampleLevelMetric, MetricCategory, MetricUseCase

class CustomEvalTask(Task):
    def __init__(self):
        super().__init__(
            name="custom_task",
            version="0.0.1",
            metrics=["accuracy", "f1"],  # Tus métricas elegidas
            description="Descripción de tu tarea de evaluación personalizada"
        )
    
    def get_prompt(self, sample):
        # Formatea tu entrada en un "prompt"
        return f"Question: {sample['question']}\nAnswer:"
    
    def process_response(self, response, ref):
        # Procesa el "output" del modelo y compáralo con la referencia
        return response.strip() == ref.strip()
```

## Métricas Personalizadas

Las tareas específicas de dominio a menudo requieren métricas especializadas. LightEval proporciona un marco flexible para crear métricas personalizadas que capturen aspectos relevantes del rendimiento del dominio:

```python
from aenum import extend_enum
from lighteval.metrics import Metrics, SampleLevelMetric, SampleLevelMetricGrouping
import numpy as np

# Definir una función de métrica a nivel de muestra
def custom_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> dict:
    """Ejemplo de métrica que genera múltiples puntuaciones por muestra"""
    response = predictions[0]
    return {
        "accuracy": response == formatted_doc.choices[formatted_doc.gold_index],
        "length_match": len(response) == len(formatted_doc.reference)
    }

# Crear una métrica que genere múltiples valores por muestra
custom_metric_group = SampleLevelMetricGrouping(
    metric_name=["accuracy", "length_match"],  # Nombres de submétricas
    higher_is_better={  # define si valores más altos son mejores para cada métrica
        "accuracy": True,
        "length_match": True
    },
    category=MetricCategory.CUSTOM,
    use_case=MetricUseCase.SCORING,
    sample_level_fn=custom_metric,
    corpus_level_fn={  # define cómo agregar cada métrica
        "accuracy": np.mean,
        "length_match": np.mean
    }
)

# Registrar la métrica en LightEval
extend_enum(Metrics, "custom_metric_name", custom_metric_group)
```

Para casos más simples donde solo necesitas un valor por muestra:

```python
def simple_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> bool:
    """Ejemplo de métrica que genera una única puntuación por muestra"""
    response = predictions[0]
    return response == formatted_doc.choices[formatted_doc.gold_index]

simple_metric_obj = SampleLevelMetric(
    metric_name="simple_accuracy",
    higher_is_better=True,
    category=MetricCategory.CUSTOM,
    use_case=MetricUseCase.SCORING,
    sample_level_fn=simple_metric,
    corpus_level_fn=np.mean  # define cómo agregar resultados entre muestras
)

extend_enum(Metrics, "simple_metric", simple_metric_obj)
```

Una vez definas tus métricas personalizadas, puedes usarlas luego en tus tareas de evaluación haciendo referencia a ellas en la configuración de la tarea. Las métricas se calcularán automáticamente en todas las muestras y se agregarán según las funciones que especifiques.

Para métricas más complejas, considera:
- Usar metadatos en tus documentos formateados para ponderar o ajustar puntuaciones
- Implementar funciones de agregación personalizadas para estadísticas a nivel de corpus
- Agregar verificaciones de validación para las entradas de tus métricas
- Documentar casos límite y comportamientos esperados

Para un ejemplo completo de métricas personalizadas en acción, consulta nuestro [proyecto de evaluación de dominio](./project/README.md).

## Creación de Conjuntos de Datos

La evaluación de alta calidad requiere conjuntos de datos cuidadosamente curados. Considera estos enfoques para la creación de conjuntos de datos:

1. **Anotación por Expertos**: Trabaja con expertos del dominio para crear y validar ejemplos de evaluación. Herramientas como [Argilla](https://github.com/argilla-io/argilla) hacen este proceso más eficiente.

2. **Datos del Mundo Real**: Recopila y anonimiza datos de uso real, asegurándote de que representen escenarios reales de despliegue del modelo.

3. **Generación Sintética**: Usa LLMs para generar ejemplos iniciales y luego permite que expertos los validen y refinen.

## Mejores Prácticas

- Documenta tu metodología de evaluación a fondo, incluidas los supuestos o limitaciones
- Incluye casos de prueba diversos que cubran diferentes aspectos de tu dominio
- Considera tanto métricas automatizadas como evaluaciones humanas donde sea apropiado
- Controla las versiones de tus conjuntos de datos y código de evaluación
- Actualiza regularmente tu conjunto de evaluaciones a medida que descubras nuevos casos límite o requisitos

## Referencias

- [Guía de Tareas Personalizadas en LightEval](https://github.com/huggingface/lighteval/wiki/Adding-a-Custom-Task)
- [Métricas Personalizadas en LightEval](https://github.com/huggingface/lighteval/wiki/Adding-a-New-Metric)
- [Documentación de Argilla](https://docs.argilla.io) para anotación de conjuntos de datos
- [Guía de Evaluación](https://github.com/huggingface/evaluation-guidebook) para principios generales de evaluación

# Próximos Pasos

⏩ Para un ejemplo completo de cómo implementar estos conceptos, consulta nuestro [proyecto de evaluación de dominio](./project/README.md).
