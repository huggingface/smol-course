# Finetuning de Modelos de Lenguaje Visual (VLM)
## Finetuning eficiente

### Cuantización
La cuantización reduce la precisión de los pesos y activaciones del modelo, lo que reduce significativamente el uso de memoria y acelera los cálculos. Por ejemplo, cambiar de `float32` a `bfloat16` reduce a la mitad los requisitos de memoria por parámetro manteniendo el rendimiento. Para una compresión más agresiva, se puede usar cuantización de 8 bits o 4 bits, reduciendo aún más el uso de memoria, aunque a costa de algo de precisión. Estas técnicas se pueden aplicar tanto a los modelos como a los ajustes del optimizador, permitiendo un entrenamiento eficiente en hardware con recursos limitados.

### PEFT y LoRA
Como se introdujo en el Módulo 3, LoRA (Low-Rank Adaptation) se centra en aprender matrices compactas de descomposición por rango, manteniendo congelados los pesos del modelo original. Esto reduce drásticamente la cantidad de parámetros entrenables, reduciendo significativamente los requisitos de recursos. LoRA, cuando se integra con PEFT, permite el Fine-tuning de modelos grandes ajustando solo un pequeño subconjunto de parámetros entrenables. Este enfoque es particularmente efectivo para adaptaciones específicas de tareas, reduciendo miles de millones de parámetros entrenables a solo millones, manteniendo el rendimiento.

### Optimización del tamaño de lote
Para optimizar el tamaño del lote para el Fine-tuning, comienza con un valor grande y redúcelo si se producen Errores de Falta de Memoria (OOM, out-of-memory). Compensa aumentando los `gradient_accumulation_steps`, manteniendo efectivamente el tamaño total del lote durante varias actualizaciones. Además, habilita `gradient_checkpointing` para reducir el uso de memoria recomputando los estados intermedios durante el pase hacia atrás, intercambiando tiempo de cálculo por requisitos de memoria de activación reducidos. Estas estrategias maximizan la utilización del hardware y ayudan a superar las limitaciones de memoria.

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./fine_tuned_model",  # Directorio para los puntos de control del modelo
    per_device_train_batch_size=4,   # Tamaño del lote por dispositivo (GPU/TPU)
    num_train_epochs=3,              # Número total de épocas de entrenamiento
    learning_rate=5e-5,              # Tasa de aprendizaje
    save_steps=1000,                 # Guarda un punto de control cada 1000 pasos
    bf16=True,                       # Usa precisión mixta para el entrenamiento
    gradient_checkpointing=True,     # Habilitado para reducir el uso de memoria de activaciones
    gradient_accumulation_steps=16,  # Acumula gradientes durante 16 pasos
    logging_steps=50                 # Registra métricas cada 50 pasos
)
```

## **Fine-tuning supervisado (SFT)**

El Fine-tuning Supervisado (SFT) adapta un modelo de lenguaje visual preentrenado (VLM) a tareas específicas utilizando conjuntos de datos etiquetados que contienen entradas emparejadas, como imágenes y texto correspondiente. Este método mejora la capacidad del modelo para realizar funciones específicas de dominio o tarea, como responder preguntas visuales, generación de descripciones de imágenes o interpretación de gráficos.

### **Visión general**
SFT es esencial cuando necesitas que un VLM se especialice en un dominio particular o resuelva problemas específicos en los que las capacidades generales del modelo base pueden no ser suficientes. Por ejemplo, si el modelo tiene dificultades con características visuales únicas o terminología específica del dominio, SFT permite que se enfoque en estas áreas aprendiendo de los datos etiquetados.

Aunque SFT es altamente efectivo, tiene limitaciones notables:
- **Dependencia de datos**: Se requieren conjuntos de datos etiquetados de alta calidad adaptados a la tarea.
- **Recursos computacionales**: Ajustar modelos grandes de VLM requiere muchos recursos.
- **Riesgo de sobreajuste (Overfitting)**: Los modelos pueden perder sus capacidades de generalización si se ajustan demasiado específicamente.

A pesar de estos desafíos, SFT sigue siendo una técnica robusta para mejorar el rendimiento del modelo en contextos específicos.

### **Uso**
1. **Preparación de los datos**: Comienza con un conjunto de datos etiquetado que empareje imágenes con texto, como preguntas y respuestas. Por ejemplo, en tareas como el análisis de gráficos, el conjunto de datos `HuggingFaceM4/ChartQA` incluye imágenes de gráficos, consultas y respuestas concisas.

2. **Configuración del modelo**: Carga un VLM preentrenado adecuado para la tarea, como `HuggingFaceTB/SmolVLM-Instruct`, y un procesador para preparar las entradas de texto e imagen. Configura el modelo para aprendizaje supervisado y adecuarlo a tu hardware.

3. **Proceso del Fine-tuning**:
   - **Formato de los datos**: Estructura el conjunto de datos en un formato similar al de un chatbot, emparejando mensajes del sistema, consultas de los usuarios y respuestas correspondientes.
   - **Configuración de entrenamiento**: Utiliza herramientas como `TrainingArguments` de Hugging Face o `SFTConfig` de TRL para configurar los parámetros de entrenamiento. Estos incluyen el tamaño de lote, la tasa de aprendizaje y los pasos de acumulación de gradientes para optimizar el uso de recursos.
   - **Técnicas de optimización**: Usa **gradient checkpointing** para ahorrar memoria durante el entrenamiento. Utiliza modelos cuantizados para reducir los requisitos de memoria y acelerar los cálculos.
   - Emplea el entrenador `SFTTrainer` de la biblioteca TRL para agilizar el proceso de entrenamiento.

## Optimización de preferencias

La optimización de preferencias, en particular la Optimización Directa de Preferencias (DPO), entrena un modelo de lenguaje visual (VLM) para alinearse con las preferencias humanas. En lugar de seguir estrictamente instrucciones predefinidas, el modelo aprende a priorizar salidas que los humanos prefieren subjetivamente. Este enfoque es particularmente útil para tareas que involucran juicio creativo, razonamiento matizado o respuestas aceptables variables.

### **Visión general**
La optimización de preferencias aborda escenarios donde las preferencias humanas subjetivas son clave para el éxito de la tarea. Al ajustar el modelo en conjuntos de datos que codifican preferencias humanas, DPO mejora la capacidad del modelo para generar respuestas que estén alineadas contextualmente y estilísticamente con las expectativas del usuario. Este método es particularmente efectivo para tareas como escritura creativa, interacciones con clientes o escenarios de opción múltiple.

A pesar de sus beneficios, la Optimización de Preferencias enfrenta desafíos:
- **Calidad de los datos**: Se requieren conjuntos de datos de alta calidad anotados con preferencias, lo que a menudo hace que la recopilación de datos sea un cuello de botella.
- **Complejidad**: El entrenamiento puede involucrar procesos sofisticados, como muestreo por pares de preferencias y balanceo de recursos computacionales.

Los conjuntos de datos de preferencias deben capturar claras preferencias entre respuestas candidatas. Por ejemplo, un conjunto de datos puede emparejar una pregunta con dos respuestas: una preferida y la otra menos aceptable. El modelo aprende a predecir la respuesta preferida, incluso si no es completamente correcta, siempre que esté mejor alineada con el juicio humano.

### **Uso**
1. **Preparación del conjunto de datos**  
   Un conjunto de datos etiquetado con preferencias es crucial para el entrenamiento. Cada ejemplo generalmente consta de un prompt (por ejemplo, una imagen y una pregunta) y dos respuestas candidatas: una elegida (preferida) y una rechazada. Por ejemplo:

   - **Pregunta**: ¿Cuántas familias?  
     - **Rechazada**: La imagen no proporciona información sobre familias.  
     - **Elegida**: La imagen muestra una tabla de Organización Unitaria con 18,000 familias.  

   El conjunto de datos enseña al modelo a priorizar respuestas mejor alineadas, incluso si no son perfectas.

2. **Configuración del modelo**  
   Carga un VLM preentrenado e intégralo con la biblioteca TRL de Hugging Face, que admite DPO, y un procesador para preparar las entradas de texto e imagen. Configura el modelo para aprendizaje supervisado y adecuarlo a tu hardware.

3. **Proceso de entrenamiento**  
   El entrenamiento implica configurar los parámetros específicos de DPO. A continuación, se resume el proceso:

   - **Formato del conjunto de datos**: Estructura cada muestra con prompts, imágenes y respuestas candidatas.
   - **Función de pérdida**: Usa una función de pérdida basada en preferencias para optimizar el modelo para seleccionar la salida preferida.
   - **Entrenamiento eficiente**: Combina técnicas como cuantización, acumulación de gradientes y adaptadores LoRA para optimizar la memoria y los cálculos.

## Recursos

- [Hugging Face Learn: Fine-tuning supervisado de VLMs](https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl)  
- [Hugging Face Learn: Fine-tuning supervisado de SmolVLM](https://huggingface.co/learn/cookbook/fine_tuning_smol_vlm_sft_trl)  
- [Hugging Face Learn: Optimización de preferencias en SmolVLM](https://huggingface.co/learn/cookbook/fine_tuning_vlm_dpo_smolvlm_instruct)  
- [Hugging Face Blog: Optimización de preferencias para VLMs](https://huggingface.co/blog/dpo_vlm)

##

 Ejemplo de código

```python
from transformers import Trainer, TrainingArguments

# Usa DPO
training_args = TrainingArguments(
    output_dir='./fine_tuned_model',
    evaluation_strategy="steps",
    save_steps=1000,
    logging_dir='./logs',
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    fp16=True,  # Usa precisión mixta
    gradient_checkpointing=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
```