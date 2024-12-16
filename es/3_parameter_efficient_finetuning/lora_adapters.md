# LoRA (Adaptación de Bajo Rango) 

LoRA (Low-Rank Adaptation) se ha convertido en el método PEFT más utilizado. Su funcionamiento se basa en añadir matrices de descomposición de rango reducido a los pesos de atención, lo que suele reducir los parámetros entrenables en un 90%.

## Entendiendo LoRA

LoRA es una técnica de fine-tuning eficiente en parámetros que congela los pesos del modelo preentrenado e inyecta matrices entrenables de descomposición de rango reducido en las capas del modelo. En lugar de entrenar todos los parámetros del modelo durante el fine-tuning, LoRA descompone las actualizaciones de los pesos en matrices más pequeñas mediante descomposición de rango, reduciendo significativamente la cantidad de parámetros entrenables mientras mantiene el rendimiento del modelo. Por ejemplo, se ha aplicado a GPT-3 175B. LoRA redujo los parámetros entrenables 10 000 veces y los requerimientos de memoria de GPU 3 veces en comparación con el fine-tuning completo. Puedes leer más sobre LoRA en el [paper de LoRA](https://arxiv.org/pdf/2106.09685).

LoRA funciona añadiendo pares de matrices de descomposición de rango reducido a las capas del transformador, normalmente enfocándose en los pesos de atención. Durante la inferencia, estos pesos adaptadores pueden fusionarse con el modelo base, lo que elimina la sobrecarga de latencia adicional. LoRA es especialmente útil para adaptar modelos de lenguaje grandes a tareas o dominios específicos mientras se mantienen requisitos de recursos manejables.

## Cargando adaptadores LoRA

Se pueden cargar adaptadores en un modelo preentrenado utilizando `load_adapter()`, lo cual es útil para probar diferentes adaptadores cuyos pesos no están fusionados. Se pueden establecer los pesos de los adaptadores activos con la función `set_adapter()`. Para devolver el modelo base, se puede usar `unload()` para descargar todos los módulos de LoRA. Esto permite cambiar fácilmente entre diferentes pesos específicos de tareas.

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("<base_model_name>")
peft_model_id = "<peft_adapter_id>"
model = PeftModel.from_pretrained(base_model, peft_model_id)
```

![lora_load_adapter](./images/lora_adapter.png)

## Fusionando adaptadores LoRA

Después de entrenar con LoRA, podrías querer fusionar los pesos del adaptador de vuelta en el modelo base para facilitar su implementación. Esto crea un único modelo con los pesos combinados, eliminando la necesidad de cargar adaptadores por separado durante la inferencia.

El proceso de fusión requiere atención al manejo de memoria y precisión. Como necesitarás cargar tanto el modelo base como los pesos del adaptador simultáneamente, asegúrate de tener suficiente memoria GPU/CPU disponible. Usar `device_map="auto"` en `transformers` ayudará con el manejo automático de memoria. Mantén una precisión consistente (por ejemplo, `float16`) durante todo el proceso, y guarda el modelo fusionado en el mismo formato para su implementación. Antes de implementar, siempre valida el modelo fusionado comparando sus salidas y métricas de rendimiento con la versión basada en adaptadores.

Los adaptadores también son convenientes para cambiar entre diferentes tareas o dominios. Puedes cargar el modelo base y los pesos del adaptador por separado, permitiendo un cambio rápido entre diferentes pesos específicos de tareas.

## Guía de implementación

El directorio `notebooks/` contiene tutoriales prácticos y ejercicios para implementar diferentes métodos PEFT. Comienza con `load_lora_adapter_example.ipynb` para una introducción básica, y luego explora `lora_finetuning.ipynb` para un análisis más detallado de cómo realizar fine-tuning en un modelo con LoRA y SFT.

Cuando implementes métodos PEFT, comienza con valores de rango pequeños (4-8) para LoRA y monitoriza la pérdida durante el entrenamiento. Utiliza conjuntos de validación para evitar el sobreajuste y compara los resultados con los baselines del fine-tuning completo cuando sea posible. La efectividad de los diferentes métodos puede variar según la tarea, por lo que la experimentación es clave.

## OLoRA

[OLoRA](https://arxiv.org/abs/2406.01775) utiliza descomposición QR para inicializar los adaptadores LoRA. OLoRA traduce los pesos base del modelo por un factor de sus descomposiciones QR, es decir, muta los pesos antes de realizar cualquier entrenamiento sobre ellos. Este enfoque mejora significativamente la estabilidad, acelera la velocidad de convergencia y logra un rendimiento superior.

## Usando TRL con PEFT

Los métodos PEFT pueden combinarse con TRL (Transformers Reinforcement Learning) para un fine-tuning eficiente. Esta integración es particularmente útil para RLHF (Reinforcement Learning from Human Feedback), ya que reduce los requisitos de memoria.

```python
from peft import LoraConfig
from transformers import AutoModelForCausalLM

# Carga el modelo con configuración PEFT
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Carga el modelo en un dispositivo específico
model = AutoModelForCausalLM.from_pretrained(
    "your-model-name",
    load_in_8bit=True,  # Opcional: usa la precisión de 8 bits
    device_map="auto",
    peft_config=lora_config
)
```

Aquí, usamos `device_map="auto"` para asignar automáticamente el modelo al dispositivo correcto. También puedes asignar manualmente el modelo a un dispositivo específico usando `device_map={"": device_index}`. Además, podrías escalar el entrenamiento en múltiples GPUs mientras mantienes un uso eficiente de memoria.

## Implementación básica de fusión

Después de entrenar un adaptador LoRA, puedes fusionar los pesos del adaptador de vuelta en el modelo base. Aquí tienes cómo hacerlo:

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 1. Carga el modelo base
base_model = AutoModelForCausalLM.from_pretrained(
    "base_model_name",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. Carga el modelo PEFT con adaptador
peft_model = PeftModel.from_pretrained(
    base_model,
    "path/to/adapter",
    torch_dtype=torch.float16
)

# 3. Fusiona los pesos del adaptador con el modelo base
try:
    merged_model = peft_model.merge_and_unload()
except RuntimeError as e:
    print(f"Fusión fallida: {e}")
    # Implementa estrategia de respaldo u optimización de memoria

# 4. Guarda el modelo fusionado
merged_model.save_pretrained("path/to/save/merged_model")
```

Si encuentras discrepancias de tamaño en el modelo guardado, asegúrate de guardar también el tokenizer:

```python
# Guarda tanto el modelo como el tokenizer
tokenizer = AutoTokenizer.from_pretrained("base_model_name")
merged_model.save_pretrained("path/to/save/merged_model")
tokenizer.save_pretrained("path/to/save/merged_model")
```

## Próximos pasos

⏩ Pasa al [Prompt Tuning](prompt_tuning.md) para aprender a realizar fine-tuning en un modelo con prompt tuning.  
⏩ Revisa el [Tutorial de Carga de Adaptadores LoRA](./notebooks/load_lora_adapter.ipynb) para aprender a cargar adaptadores LoRA.

# Recursos

- [LORA: ADAPTACIÓN DE BAJO RANGO PARA MODELOS DE LENGUAJE DE GRAN TAMAÑO](https://arxiv.org/pdf/2106.09685)  
- [Documentación de PEFT](https://huggingface.co/docs/peft)  
- [Publicación en el blog de Hugging Face sobre PEFT](https://huggingface.co/blog/peft)  