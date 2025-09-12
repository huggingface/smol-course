# Ajuste por Instrucciones (Instruction Tuning)

Este módulo te guiará en el proceso de ajuste por instrucciones de modelos de lenguaje. El ajuste por instrucciones implica adaptar modelos preentrenados a tareas específicas mediante un entrenamiento adicional sobre conjuntos de datos específicos para esas tareas. Este proceso ayuda a los modelos a mejorar su rendimiento en tareas específicas.

En este módulo, exploraremos dos temas: 1) Plantillas de Chat y 2) Fine-tuning Supervisado

## 1️⃣ Plantillas de Chat

Las plantillas de chat estructuran las interacciones entre los usuarios y los modelos de IA, asegurando respuestas coherentes y contextualmente apropiadas. Estas plantillas incluyen componentes como los mensajes de sistema y mensajes basados en roles. Para más información detallada, consulta la sección de [Plantillas de Chat](./chat_templates.md).

## 2️⃣ Fine-tuning Supervisado

El Fine-tuning supervisado (SFT) es un proceso crítico para adaptar modelos de lenguaje preentrenados a tareas específicas. Implica entrenar el modelo en un conjunto de datos específico para la tarea con ejemplos etiquetados. Para una guía detallada sobre el SFT, incluyendo pasos clave y mejores prácticas, consulta la página de [Fine-tuning Supervisado](./supervised_fine_tuning.md).

## Referencias

- [Documentación de Transformers sobre plantillas de chat](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Script para Fine-Tuning Supervisado en TRL](https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py)
- [`SFTTrainer` en TRL](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [Papel de Optimización Directa de Preferencias](https://arxiv.org/abs/2305.18290)
- [Fine-Tuning Supervisado con TRL](https://huggingface.co/docs/trl/main/en/tutorials/supervised_finetuning)
- [Cómo afinar Google Gemma con ChatML y Hugging Face TRL](https://www.philschmid.de/fine-tune-google-gemma)
- [Fine-Tuning de LLM para generar catálogos de productos persas en formato JSON](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_to_generate_persian_product_catalogs_in_json_format)
