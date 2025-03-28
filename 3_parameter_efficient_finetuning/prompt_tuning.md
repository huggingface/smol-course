# Prompt Tuning

Prompt tuning 也是一个高效的微调手段，不同于微调模型参数，它改变的是输入模型的表征。具体来说，prompt tuning 在训练前会添加几个额外的 token，训练过程中，token 的 embedding 被更新，而模型参数一直保持不变。

## 理解 Prompt Tuning

Prompt tuning 通常把可以训练的连续向量（也称为soft prompts）接在输入文本的前面。与离散的文本提示词不同，这些 soft prompts 是通过训练过程中经反向传播更新的，而语言模型则在训练中不变。该方法在 [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691) 中提出，展示了其在模型尺寸增大时的有效性：当模型尺寸在 10B 参数量左右时，prompt tuning 只更新 soft prompts 的几百个参数，即可达到模型全量微调的效果。

Soft prompts 是模型的 embedding 空间中的一些连续数值的向量，它们会在微调过程中被更新。传统的 prompts 是一些离散的 tokens，在自然语言层面代表某些语义信息；soft prompt 则没有这些内在含义，经过参数更新后，它被用来从模型中引出一些特定行为。这种方法在多任务领域尤其有效，因为每个任务仅需保存一个 soft prompt 的向量（通常仅几百个参数），而不是对整个模型复制。这种做法不仅现存占用少，而且还能支持快速的任务切换，无需模型重新加载。

## 训练过程

Soft prompts 一般包含 8 到 32 个 tokens，它的初始化可以是随机初始化，也可以是来自现有的文本。初始化过程很影响训练，后者的方法通常效果更好。

训练过程中，仅这些 soft prompts 的参数会被更新；训练用的损失函数也没有变化；但学习率需要我们认真调节，同时也建议观察 soft prompts 上面的梯度信息，以免训练失败。

## 基于 PEFT 的代码实现

使用 PEFT 库实现 prompt tuning 非常简单直接，以下是一个简单的例子：

```python
from peft import PromptTuningConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
model = AutoModelForCausalLM.from_pretrained("your-base-model")
tokenizer = AutoTokenizer.from_pretrained("your-base-model")

# Configure prompt tuning
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=8,  # soft prompts 的 token 数量
    prompt_tuning_init="TEXT",  # 初始化方法为从文本初始化
    prompt_tuning_init_text="Classify if this text is positive or negative:",
    tokenizer_name_or_path="your-base-model",
)

# Create prompt-tunable model
model = get_peft_model(model, peft_config)
```

## 与其它方法的对比

对比与其它 PEFT 方法，prompt tuning 胜在它的高效性。虽然 LoRA 也减少了训练参数以及所需显存，但反复加载 adapter 来切换任务就很麻烦。Prompt tuning 需要的训练参数更少，且任务切换更加方便。而全量参数微调则既需要超大的训练资源，也需要通过全量参数重新载入来切换任务。

| 方法 | 训练参数量 | 显存需求 | 任务切换难度 |
|--------|------------|---------|----------------|
| Prompt Tuning | 很低 | 很低 | 非常简单 |
| LoRA | 低 | 低 | 需要加载 Adapter |
| 全量参数微调 | 很高 | 很高 | 加载所有参数 |

在实际 prompt tuning 训练过程中，建议先从较小的 virtual tokens 数量开始（如 8 到 16 个），仅当任务复杂度增加时，再增加 virtual tokens 数量。从文本初始化通常比随机初始化更好，尤其是你使用和任务相关的文本时。初始化方法需要和你的任务复杂度匹配。

训练过程中，你还需注意学习率。如果使用较大的学习率，你需要时刻观察 soft prompt 更新时的梯度信息，以防训练崩溃。训练过程中，定期的验证也是确保性能的良好手段。

## 应用

Prompt tuning 在这些场景下优势明显：

1. 多任务下的大语言模型部署
2. 计算资源有限的训练场景
3. 需要在不同任务间快速切换的场景
4. 针对隐私敏感的应用场景

而当模型变小时，prompt tuning 就没有那么有竞争力了。比如在 SmolLM2 这样的模型尺寸下，prompt tuning 的意义就不大，可能还不如全量微调。

## 接下来

⏭️ 学习 [LoRA Adapters 的教程](./notebooks/finetune_sft_peft.ipynb)，了解如何实操 LoRA 微调模型的过程。

## 学习资源
- [PEFT 官方文档](https://huggingface.co/docs/peft)
- [Prompt Tuning 论文](https://arxiv.org/abs/2104.08691)
- [Hugging Face Cookbook 中关于 prompt tuning 的部分](https://huggingface.co/learn/cookbook/prompt_tuning_peft)
