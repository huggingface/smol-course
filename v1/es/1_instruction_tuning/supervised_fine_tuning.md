# Fine-Tuning Supervisado

El Fine-Tuning Supervisado (SFT) es un proceso crítico para adaptar modelos de lenguaje preentrenados a tareas o dominios específicos. Aunque los modelos preentrenados tienen capacidades generales impresionantes, a menudo necesitan ser personalizados para sobresalir en casos de uso particulares. El SFT cubre esta brecha entrenando el modelo con conjuntos de datos cuidadosamente seleccionados que contienen ejemplos validados por humanos.

## Comprendiendo el Fine-Tuning Supervisado

En su núcleo, el Fine-Tuning supervisado se trata de enseñar a un modelo preentrenado a realizar tareas específicas mediante ejemplos de tokens etiquetados. El proceso consiste en mostrarle al modelo muchos ejemplos del comportamiento deseado de entrada-salida, permitiéndole aprender los patrones específicos de tu caso de uso.

El SFT es eficaz porque utiliza el conocimiento fundamental adquirido durante el preentrenamiento, mientras adapta el comportamiento del modelo para que se ajuste a tus necesidades específicas.

## Cuándo Utilizar Fine-Tuning Supervisado

La decisión de usar SFT generalmente depende de la brecha entre las capacidades actuales de tu modelo y tus requisitos específicos. El SFT se vuelve particularmente valioso cuando necesitas un control preciso sobre las salidas del modelo o cuando trabajas en dominios especializados.

Por ejemplo, si estás desarrollando una aplicación de atención al cliente, es posible que desees que tu modelo siga consistentemente las pautas de la empresa y maneje consultas técnicas de una manera estandarizada. De manera similar, en aplicaciones médicas o legales, la precisión y la adherencia a la terminología específica del dominio se vuelven cruciales. En estos casos, el SFT puede ayudar a alinear las respuestas del modelo con los estándares profesionales y la experiencia del dominio.

## El Proceso de Fine-Tuning Supervisado (SFT)

El proceso de Fine-Tuning Supervisado consiste en entrenar los pesos del modelo en un conjunto de datos específico para una tarea.

Primero, necesitarás preparar o seleccionar un conjunto de datos que represente tu tarea objetivo. Este conjunto de datos debe incluir ejemplos diversos que cubran una amplia gama de escenarios que tu modelo encontrará. La calidad de estos datos es importante: cada ejemplo debe demostrar el tipo de salida que deseas que el modelo produzca. Luego, viene la fase de fine-tuning, donde usarás marcos como `transformers` y `trl` de Hugging Face para entrenar el modelo con tu conjunto de datos.

Durante todo el proceso, la evaluación continua es esencial. Querrás monitorear el rendimiento del modelo en un conjunto de validación para asegurarte de que está aprendiendo los comportamientos deseados sin perder sus capacidades generales. En el [módulo 4](../4_evaluation), cubriremos cómo evaluar tu modelo.

## El Rol del SFT en la Alineación con las Preferencias

El SFT juega un papel fundamental en la alineación de los modelos de lenguaje con las preferencias humanas. Técnicas como el Aprendizaje por Refuerzo a partir de Retroalimentación Humana (RLHF) y la Optimización Directa de Preferencias (DPO) dependen del SFT para formar un nivel base de comprensión de la tarea antes de alinear aún más las respuestas del modelo con los resultados deseados. Los modelos preentrenados, a pesar de su competencia general en el lenguaje, no siempre generan salidas que coinciden con las preferencias humanas. El SFT cubre esta brecha al introducir datos específicos de dominio y orientación, lo que mejora la capacidad del modelo para generar respuestas que se alineen más estrechamente con las expectativas humanas.

## Fine-Tuning Supervisado con Aprendizaje por Refuerzo de Transformers (TRL)

Un paquete de software clave para el fine-tuning supervisado es el Aprendizaje por Refuerzo de Transformers (TRL). TRL es un conjunto de herramientas utilizado para entrenar modelos de lenguaje transformadores utilizando aprendizaje por refuerzo (RL).

Basado en la librería Transformers de Hugging Face, TRL permite a los usuarios cargar directamente modelos de lenguaje preentrenados y es compatible con la mayoría de las arquitecturas de decodificadores y codificadores-decodificadores. La librería facilita los principales procesos del RL utilizado en la modelización del lenguaje, incluyendo el fine-tuning supervisado (SFT), la modelización de recompensas (RM), la optimización de políticas proximales (PPO) y la optimización directa de preferencias (DPO). Usaremos TRL en varios módulos a lo largo de este repositorio.

# Próximos Pasos

Prueba los siguientes tutoriales para obtener experiencia práctica con el SFT utilizando TRL:

⏭️ [Tutorial de Plantillas de Chat](./notebooks/chat_templates_example.ipynb)

⏭️ [Tutorial de Fine-Tuning Supervisado](./notebooks/supervised_fine_tuning_tutorial.ipynb)
