# LoRA（低秩分解）

LoRA 是最常用的 PEFT 算法。它针对 attention 层的权重加入参数量较少、低秩分解过的参数矩阵，用模型原有参数和低秩分解参数计算出的激活值之和代表微调过后的激活值。这样当我们只更新低秩分解过的参数矩阵时，我们需要训练的参数量能减少大约 90%。

## 理解 LoRA

LoRA 全称是 Low-Rank Adaptation，或叫做“低秩分解”。它的基本做法是，在微调时，冻结所有预训练模型的参数，同时为需要微调的模型层注入额外的可训练的参数矩阵（通常称之为 Adapter）。通过对需要微调的层的参数矩阵进行低秩分解，可以得到两个参数量较小的新参数矩阵；而这一层的前向计算激活值则可以用“原有参数矩阵计算出的激活值”加上“低秩分解出的两个矩阵计算出的激活值”而得到。训练时，我们只需要训练两个低秩分解的矩阵即可，这样极大减少了所需微调的参数量，同时也能保持原有模型性能。例如，如果我们用 LoRA 微调 GPT-3 175B 模型，相比于全量参数的微调，LoRA 需要参与训练的参数量可减少至万分之一、GPU 现存需求可减少至三分之一。感兴趣的读者可以阅读 [LoRA 的论文](https://arxiv.org/pdf/2106.09685)。

一般而言，LoRA 都是对 transformer 层的参数进行低秩分解的，尤其是在与注意力机制相关的参数上。在推理过程中，这些 adapter 的参数可以被直接融合进模型中，得到与原模型结构完全一致的新模型，无需增加新的层。得益于此，LoRA 尤其适合在低计算资源情况下，将大模型适配进入特定的任务领域。

## 如何载入 LoRA 的 Adapters

如果你使用 `peft` 库，你可以用 `load_adapter()` 载入 LoRA Adapters。这对你尝试不同的 adapter 非常管用，因为它还不会将参数融合进原模型。你可以使用 `set_adapter()` 指定哪个 LoRA Adapter 在生效。如果想返回原模型，你可以使用 `unload()` 卸载所有 LoRA 参数。这种设定使得在不同任务领域间切换模型变得非常容易。

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("<base_model_name>")
peft_model_id = "<peft_adapter_id>"
model = PeftModel.from_pretrained(base_model, peft_model_id)
```

![lora_load_adapter](./images/lora_adapter.png)

## 将 LoRA 参数融入原模型

如果在 LoRA 微调结束后，你想直接获取一套新的模型参数，而不是每次使用的时候都需要加载 LoRA 的 Adapter，你可以直接将 LoRA 的参数融入原模型中。

融合的时候，我们首先需要注意内存的管理以及参数的精度。因为我们要同时载入原模型和 LoRA 参数，需要注意 GPU 或 CPU 的内存是否够用。在 `transformers` 中使用 `device_map="auto"` 可以替我们自动进行内存管理。同时，要注意原模型、LoRA 参数的精度需保持一致。融合后，检查模型输出是否和未融合是一致也很重要。

## 代码实现

在 `notebooks/` 目录下，有 PEFT 相关方法的实践教程以及练习题。我们首先会在 `load_lora_adapter_example.ipynb` 学习加载 LoRA Adapter 相关的内容，然后在 `lora_finetuning.ipynb` 中，我们将学习如果用 LoRA 进行 SFT。

一个比较好的 LoRA 训练流程应该是，首先从较低的秩开始，一般是 4 到 8，同时观察训练损失值。使用验证集及时查看避免过拟合。不同任务可能有较大差异，所有还是要以实际实验现象为准。

## OLoRA

[OLoRA](https://arxiv.org/abs/2406.01775) 使用 QR 分解来初始化 LoRA 的 Adapter。该算法对原有的参数矩阵 W 分解为 Q 和 R 两个矩阵，其中 Q 矩阵包含 W 矩阵的 r 个正交向量，使得优化能够在一个较好的子空间进行。这样可以很大地提升收敛速度，同时也达到了非常好的效果。

##  TRL 与 PEFT 结合使用

PEFT 也可以和 TRL 库一起使用，这对 RLHF（Reinforcement Learning from Human Feedback）尤其实用。

```python
from peft import LoraConfig
from transformers import AutoModelForCausalLM

# Load model with PEFT config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Load model on specific device
model = AutoModelForCausalLM.from_pretrained(
    "your-model-name",
    load_in_8bit=True,  # Optional: use 8-bit precision
    device_map="auto",
    peft_config=lora_config
)
```

在上述代码中，我们用 `device_map="auto"` 自动分配模型到正确的计算设备上。关于具体计算设备，你也可以手动修改：`device_map={"": device_index}`；当然你也可以扩大训练规模，如实用多 GPU 训练等。

## 基本的参数融合实现



训练好 LoRA adapter 后，将权重融合回原模型的方法如下：


```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 1. Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    "base_model_name",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. Load the PEFT model with adapter
peft_model = PeftModel.from_pretrained(
    base_model,
    "path/to/adapter",
    torch_dtype=torch.float16
)

# 3. Merge adapter weights with base model
try:
    merged_model = peft_model.merge_and_unload()
except RuntimeError as e:
    print(f"Merging failed: {e}")
    # Implement fallback strategy or memory optimization

# 4. Save the merged model
merged_model.save_pretrained("path/to/save/merged_model")
```

保存的时候，你可能也需要保存 tokenizer 到相应目录。

```python
# Save both model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("base_model_name")
merged_model.save_pretrained("path/to/save/merged_model")
tokenizer.save_pretrained("path/to/save/merged_model")
```

## 接下来

⏩ 继续学习 [Prompt Tuning](prompt_tuning_cn.md)，了解这种微调方式如何运作。

⏩ 实践 [加载 LoRA Adapters 的教程](./notebooks/load_lora_adapter_cn.ipynb) 练习加载 LoRA adapters。

# 学习资源

- [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685)
- [PEFT 官方文档](https://huggingface.co/docs/peft)
- [Hugging Face 有关 PEFT 的博客](https://huggingface.co/blog/zh/peft)
