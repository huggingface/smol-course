# Fine-Tuning de Modelos de Visión-Lenguaje

## Fine-Tuning Eficiente

### Cuantización
La cuantización reduce la precisión de los pesos y activaciones del modelo, disminuyendo significativamente el uso de memoria y acelerando los cálculos. Por ejemplo, cambiar de `float32` a `bfloat16` reduce a la mitad los requisitos de memoria por parámetro manteniendo el rendimiento. Para una compresión más agresiva, se pueden usar cuantizaciones de 8 bits y 4 bits, reduciendo aún más el uso de memoria, aunque con cierta pérdida de precisión. Estas técnicas pueden aplicarse tanto al modelo como a los ajustes del optimizador, permitiendo entrenamientos eficientes en hardware con recursos limitados.

### PEFT & LoRA
Como se introdujo en el Módulo 3, LoRA (Low-Rank Adaptation) se enfoca en aprender matrices de descomposición de rango reducido mientras se mantienen congelados los pesos originales del modelo. Esto reduce drásticamente la cantidad de parámetros entrenables, disminuyendo significativamente la cantidad de recursos requeridos. LoRA, cuando se integra con PEFT, permite el fine-tuning de modelos grandes ajustando solo un subconjunto pequeño y entrenable de parámetros. Este enfoque es particularmente efectivo para adaptaciones a tareas específicas, reduciendo miles de millones de parámetros entrenables a solo millones, manteniendo el mismo rendimiento.

### Optimización del Tamaño del Lote (Batch Size en inglés)
Para optimizar el tamaño del lote en el proceso de fine-tuning, comienza con un valor grande y redúcelo si ocurren errores de memoria (OOM). Compensa aumentando `gradient_accumulation_steps`, lo que mantiene el tamaño total del lote en múltiples actualizaciones. Además, habilita `gradient_checkpointing` para reducir el uso de memoria mediante el recálculo de estados intermedios durante la propogación hacia atrás, intercambiando tiempo de cómputo por menor uso de memoria de activaciones. Estas estrategias maximizan la utilización del hardware y ayudan a superar las restricciones de memoria.

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./fine_tuned_model",  # Directorio para checkpoints
    per_device_train_batch_size=4,    # Tamaño del lote por dispositivo (GPU/TPU)
    num_train_epochs=3,               # Número total de épocas de entrenamiento
    learning_rate=5e-5,               # Tasa de aprendizaje
    save_steps=1000,                  # Guardar checkpoint cada 1000 pasos
    bf16=True,                        # Usar precisión mixta para el entrenamiento
    gradient_checkpointing=True,      # Habilitar para reducir el uso de memoria de activaciones
    gradient_accumulation_steps=16,   # Acumular gradientes en 16 pasos
    logging_steps=50                  # Registrar métricas cada 50 pasos
)
```

## **Supervised Fine-Tuning (SFT)**

El Supervised Fine-Tuning (SFT) adapta un Modelo de Visión-Lenguaje (VLM) preentrenado a tareas específicas mediante el uso de conjuntos de datos etiquetados que contienen pares de imágenes y texto. Este método mejora la capacidad del modelo para desempeñar funciones específicas en un dominio o tarea, como responder a preguntas visuales, generar descripciones de imágenes o interpretar gráficos.

### **Descripción General**
SFT es esencial cuando se necesita que un VLM se especialice en un dominio o resuelva problemas específicos donde las capacidades generales del modelo base pueden ser insuficientes. Por ejemplo, si el modelo tiene dificultades con características visuales únicas o terminología especializada, SFT le permite enfocarse en estas áreas mediante el aprendizaje con datos etiquetados.

Si bien el SFT es altamente efectivo, presenta ciertas limitaciones:
- **Dependencia de Datos**: Se requieren conjuntos de datos etiquetados de alta calidad y adecuados para la tarea.
- **Recursos Computacionales**: El fine-tuning de grandes VLMs consume muchos recursos.
- **Riesgo de Sobreajuste**: Si el fine-tuning es demasiado ajustado a los datos, el modelo puede perder su capacidad de generalización.

A pesar de estos desafíos, SFT sigue siendo una técnica robusta para mejorar el rendimiento del modelo en contextos específicos.

### **Uso**
1. **Preparación de Datos**: Comienza con un conjunto de datos etiquetado que relacione imágenes con texto, como preguntas y respuestas. Por ejemplo, en tareas como el análisis de gráficos, el conjunto `HuggingFaceM4/ChartQA` incluye imágenes de gráficos, consultas y respuestas concisas.

2. **Configuración del Modelo**: Carga un VLM preentrenado adecuado para la tarea, como `HuggingFaceTB/SmolVLM-Instruct`, y un procesador para preparar entradas de texto e imagen. Configura el modelo para aprendizaje supervisado y adáptalo a tu hardware.

3. **Proceso de Fine-Tuning**:
   - **Formateo de Datos**: Estructura el conjunto de datos en un formato de chatbot, emparejando mensajes del sistema, consultas de usuario y respuestas correspondientes.
   - **Configuración del Entrenamiento**: Usa herramientas como `TrainingArguments` de Hugging Face o `SFTConfig` de TRL para configurar los parámetros del entrenamiento, como el tamaño del lote, la tasa de aprendizaje y los pasos de acumulación de gradientes.
   - **Técnicas de Optimización**: Usa **gradient checkpointing** para ahorrar memoria durante el entrenamiento y emplea modelos cuantizados para reducir los requisitos de memoria y acelerar los cálculos.
   - Emplea `SFTTrainer` de la librería TRL para simplificar el proceso de entrenamiento.

## Optimización por Preferencias

La Optimización por Preferencias, en particular la Direct Preference Optimization (DPO), entrena un Modelo de Visión-Lenguaje (VLM) para alinearse con las preferencias humanas. En lugar de seguir estrictamente instrucciones predefinidas, el modelo aprende a priorizar resultados que los humanos consideran preferibles. Este enfoque es útil para tareas que requieren juicios creativos, razonamiento matizado o respuestas variadas.

### **Descripción General**
La Optimización por Preferencias es ideal en escenarios donde las preferencias humanas son clave para el éxito de la tarea. Mediante el fine-tuning en conjuntos de datos que codifican preferencias humanas, DPO mejora la capacidad del modelo para generar respuestas alineadas con las expectativas del usuario. Este método es particularmente efectivo para tareas como escritura creativa, interacciones con clientes o escenarios de opción múltiple.

Apesar de sus beneficios, este método tiene desafíos:
- **Calidad de Datos**: Se requieren conjuntos de datos anotados con preferencias de alta calidad, lo que puede ser un cuello de botella en la recolección de datos.
- **Complejidad**: El entrenamiento involucra procesos sofisticados como muestreo por pares de preferencias y balanceo de recursos computacionales.

Los conjuntos de datos de preferencias deben capturar diferencias claras entre respuestas candidatas. Por ejemplo, un dataset puede emparejar una pregunta con dos respuestas: una preferida y otra menos aceptable. El modelo aprende a predecir la respuesta preferida, incluso si no es completamente correcta, siempre que esté mejor alineada con el juicio humano.

### **Uso**  
1. **Preparación del Conjunto de Datos**  
   Un conjunto de datos etiquetado con preferencias es fundamental para el entrenamiento. Cada ejemplo suele consistir en un prompt (por ejemplo, una imagen y una pregunta) y dos respuestas candidatas: una elegida (preferida) y otra rechazada. Por ejemplo:  

   - **Pregunta**: ¿Cuántas familias?  
     - **Rechazada**: La imagen no proporciona información sobre familias.  
     - **Preferida**: La imagen muestra una tabla de una Organización Sindical con 18,000 familias.  

   El conjunto de datos enseña al modelo a priorizar respuestas mejor alineadas, incluso si no son perfectas.  

2. **Configuración del Modelo**  
   Carga un VLM preentrenado e intégralo con la biblioteca TRL de Hugging Face, que admite DPO, junto con un procesador para preparar las entradas de texto e imagen. Configura el modelo para aprendizaje supervisado y adáptalo a tu hardware.  

3. **Pipeline de Entrenamiento**  
   El entrenamiento implica la configuración de parámetros específicos de DPO. Aquí tienes un resumen del proceso:  

   - **Formateo del Conjunto de Datos**: Estructura cada muestra con prompts, imágenes y respuestas candidatas.  
   - **Función de Pérdida**: Usa una función de pérdida basada en preferencias para optimizar el modelo en la selección de la respuesta preferida.  
   - **Entrenamiento Eficiente**: Combina técnicas como cuantización, acumulación de gradientes y adaptadores LoRA para optimizar el uso de memoria y el cómputo.


## Recursos

- [Hugging Face Learn: Supervised Fine-Tuning VLMs](https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl) 
- [Hugging Face Learn: Supervised Fine-Tuning SmolVLM](https://huggingface.co/learn/cookbook/fine_tuning_smol_vlm_sft_trl)  
- [Hugging Face Blog: Preference Optimization for VLMs](https://huggingface.co/blog/dpo_vlm)

## Próximos Pasos

⏩ Prueba el [vlm_finetune_sample.ipynb](./notebooks/vlm_finetune_sample.ipynb) para implementar este enfoque de alineación de preferencias.
