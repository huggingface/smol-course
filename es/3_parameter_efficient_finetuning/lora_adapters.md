# LoRA (Adaptación de Bajo Rango)

LoRA se ha convertido en el método PEFT más ampliamente adoptado. Funciona agregando matrices de descomposición de bajo rango a los pesos de la atención, lo que generalmente reduce los parámetros entrenables en aproximadamente un 90%. Aquí tienes una configuración básica:

```python
from peft import LoraConfig

config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,  # Rango de las matrices de actualización
    lora_alpha=32,  # Escala de las actualizaciones
    lora_dropout=0.1
)
```

Esta configuración se puede utilizar con TRL para entrenar un modelo con LoRA. Las matrices de LoRA se conocen como adaptadores y pueden fusionarse nuevamente con el modelo base.

## Fusionar adaptadores LoRA

Después de entrenar con LoRA, es posible que quieras fusionar los pesos del adaptador con el modelo base para facilitar su implementación. Esto crea un modelo único con los pesos combinados, eliminando la necesidad de cargar los adaptadores por separado durante la inferencia.

El proceso de fusión requiere atención a la gestión de memoria y la precisión. Como necesitarás cargar tanto el modelo base como los pesos del adaptador simultáneamente, asegúrate de que haya suficiente memoria GPU/CPU disponible. Usar `device_map="auto"` puede ayudar con la gestión automática de memoria. Mantén una precisión consistente (por ejemplo, float16) durante todo el proceso, coincidiendo con la precisión utilizada durante el entrenamiento y guarda el modelo fusionado en el mismo formato para la implementación. Antes de implementar, siempre valida el modelo fusionado comparando sus salidas y métricas de rendimiento con la versión basada en adaptadores.

### Proceso Básico de Fusión

Aquí te mostramos cómo fusionar un adaptador LoRA de vuelta al modelo base:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 1. Cargar el modelo base
base_model = AutoModelForCausalLM.from_pretrained(
    "base_model_name",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. Cargar el modelo PEFT con el adaptador
peft_model = PeftModel.from_pretrained(
    base_model,
    "path/to/adapter",
    torch_dtype=torch.float16
)

# 3. Fusionar los pesos del adaptador con el modelo base
merged_model = peft_model.merge_and_unload()

# 4. Guardar el modelo fusionado
merged_model.save_pretrained("path/to/save/merged_model")
```

Si encuentras discrepancias en el tamaño del modelo guardado, asegúrate de guardar también el tokenizer:

```python
# Guardar tanto el modelo como el tokenizer
tokenizer = AutoTokenizer.from_pretrained("base_model_name")
merged_model.save_pretrained("path/to/save/merged_model")
tokenizer.save_pretrained("path/to/save/merged_model")
```

## Guía de Implementación

Comienza instalando los paquetes requeridos:
```bash
pip install transformers peft accelerate
```

El directorio `notebooks/` contiene tutoriales prácticos para implementar diferentes métodos PEFT. Comienza con `lora_finetuning.ipynb` para una introducción básica, luego explora ajuste de prompts y prefix a través de sus respectivos notebooks.

Al implementar métodos PEFT, comienza con valores de rango pequeños (4-8) para LoRA y monitorea la pérdida de entrenamiento. Usa conjuntos de validación para evitar el sobreajuste y compara los resultados con las bases de referencia de ajuste completo cuando sea posible. La efectividad de diferentes métodos puede variar según la tarea, por lo que la experimentación es clave.

## Usando TRL con PEFT

Los métodos PEFT se pueden combinar con TRL (Transformer Reinforcement Learning) para un ajuste eficiente de aprendizaje por refuerzo. Esta integración es particularmente útil para RLHF (Reinforcement Learning from Human Feedback) ya que reduce los requisitos de memoria.

Puedes escalar el entrenamiento a través de múltiples GPUs manteniendo el uso eficiente de la memoria. Así es como configurarlo:

```python
from accelerate import Accelerator
from peft import LoraConfig
from transformers import AutoModelForCausalLM

# Obtener el dispositivo actual del acelerador
accelerator = Accelerator()
current_device = accelerator.process_index

# Cargar el modelo con la configuración PEFT
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Cargar el modelo en el dispositivo específico
model = AutoModelForCausalLM.from_pretrained(
    "your-model-name",
    load_in_8bit=True,  # Opcional: usar precisión de 8 bits
    device_map={"": current_device},  # Asignar al dispositivo correcto
    peft_config=lora_config
)
```

## Recursos

1. LoRA: [LORA: ADAPTACIÓN DE BAJO RANGO DE GRANDES MODELOS DE LENGUAJE](https://arxiv.org/pdf/2106.09685.pdf)
2. Prefix Tuning: [P-Tuning v2: El Ajuste de Prompts Puede Ser Comparable al Ajuste Completo de Manera Universal a Través de Escalas y Tareas](https://arxiv.org/pdf/2110.07602.pdf)
3. Prompt Tuning: [El Poder de la Escala para el Ajuste de Prompts Eficiente en Parámetros](https://arxiv.org/pdf/2104.08691.pdf)
4. P-Tuning: [GPT También Entiende](https://arxiv.org/pdf/2103.10385.pdf)
- [Documentación de PEFT](https://huggingface.co/docs/peft)
- [Guía de PEFT de Hugging Face](https://huggingface.co/blog/peft)