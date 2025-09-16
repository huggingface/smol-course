# Evaluación

La evaluación es un paso crítico en el desarrollo y despliegue de modelos de lenguaje. Nos permite entender qué tan bien funcionan nuestros modelos en diferentes capacidades e identificar áreas de mejora. Este módulo cubre tanto los "becnhmarks" estándares como los enfoques de evaluación específicos para evaluar de manera integral tu modelo **smol**.

Usaremos [`lighteval`](https://github.com/huggingface/lighteval), una poderosa biblioteca de evaluación desarrollada por Hugging Face que se integra perfectamente con el ecosistema de Hugging Face. Para una explicación más detallada sobre los conceptos y mejores prácticas de evaluación, consulta la [guía de evaluación](https://github.com/huggingface/evaluation-guidebook).

## Descripción del Módulo

Una estrategia de evaluación completa examina múltiples aspectos del rendimiento del modelo. Evaluamos capacidades específicas en tareas como responder preguntas y resumir textos, para entender cómo el modelo maneja diferentes tipos de problemas. Medimos la calidad del "output" mediante factores como coherencia y precisión. A su vez, la evaluación de seguridad ayuda a identificar posibles "outputs" dañinas o sesgos. Finalmente, las pruebas de experticia en áreas especfícios verifican el conocimiento especializado del modelo en tu campo objetivo.

## Contenidos

### 1️⃣ [Evaluaciones Automáticas](./automatic_benchmarks.md)
Aprende a evaluar tu modelo utilizando "benchmarks" y métricas estandarizadas. Exploraremos "benchmarks" comunes como MMLU y TruthfulQA, entenderemos las métricas clave de evaluación y configuraciones, y cubriremos mejores prácticas para una evaluación reproducible.

### 2️⃣ [Evaluación Personalizada en un Dominio](./custom_evaluation.md)
Descubre cómo crear flujos de evaluación adaptados a tus casos de uso específicos. Te guiaremos en el diseño de tareas de evaluación personalizadas, la implementación de métricas especializadas y la construcción de conjuntos de datos de evaluación que se ajusten a tus necesidades.

### 3️⃣ [Proyecto de Evaluación en un Dominio](./project/README.md)
Sigue un ejemplo completo de cómo construir un flujo de evaluación específico para un dominio. Aprenderás a generar conjuntos de datos de evaluación, usar Argilla para la anotación de datos, crear conjuntos de datos estandarizados y evaluar modelos utilizando LightEval.

## Recursos

- [Guía de Evaluación](https://github.com/huggingface/evaluation-guidebook) - Guía completa para la evaluación de modelos de lenguaje
- [Documentación de LightEval](https://github.com/huggingface/lighteval) - Documentación oficial de la biblioteca LightEval
- [Documentación de Argilla](https://docs.argilla.io) - Aprende sobre la plataforma de anotación Argilla
- [Paper de MMLU](https://arxiv.org/abs/2009.03300) - Artículo sobre el benchmark MMLU
- [Crear una Tarea Personalizada](https://github.com/huggingface/lighteval/wiki/Adding-a-Custom-Task)
- [Crear una Métrica Personalizada](https://github.com/huggingface/lighteval/wiki/Adding-a-New-Metric)
- [Usar métricas existentes](https://github.com/huggingface/lighteval/wiki/Metric-List)
