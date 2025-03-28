# VLM 微调

## 高效微调

### 量化

量化可降低模型权重与激活值的精度，大幅减少内存使用并加快计算速度。例如，从 `float32` 转换为 `bfloat16`，每个参数的内存需求减半，同时还能维持性能。若要进行更激进的压缩，可采用 8 位和 4 位量化，进一步降低内存使用，不过这会牺牲一定的准确性。这些技术既能应用于模型，也能用于优化器设置，从而使在资源有限的硬件上高效训练成为可能。

### PEFT & LoRA

专注于学习紧凑的秩分解矩阵，同时保持原始模型的权重不变。这极大地减少了可训练参数的数量，显著降低了资源需求。当 LoRA 与 PEFT 相结合时，仅通过调整一小部分可训练参数就能对大型模型进行微调。这种方法对于特定任务的微调特别有效，它能将数十亿个可训练参数减少到仅数百万个，同时维持性能。

### Batch Size 的优化

为优化微调时的 batch size 大小，可先从较大的值开始，如果出现内存不足（OOM）错误，则将其减小。可通过增加 `gradient_accumulation_steps` 来补偿，从而在多次更新过程中有效地维持总批量大小。此外，启用 `gradient_checkpointing` 可通过在反向传播期间重新计算中间状态来降低内存使用，以计算时间为代价来减少激活内存需求。这些策略可使硬件利用率最大化，并有助于克服内存限制。

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./fine_tuned_model",  # Directory for model checkpoints
    per_device_train_batch_size=4,   # Batch size per device (GPU/TPU)
    num_train_epochs=3,              # Total training epochs
    learning_rate=5e-5,              # Learning rate
    save_steps=1000,                 # Save checkpoint every 1000 steps
    bf16=True,                       # Use mixed precision for training
    gradient_checkpointing=True,     # Enable to reduce activation memory usage
    gradient_accumulation_steps=16,  # Accumulate gradients over 16 steps
    logging_steps=50                 # Log metrics every 50 steps
)
```

## **有监督微调（SFT）**

有监督微调（Supervised Fine-tuning 或 SFT）通过利用包含配对输入（例如图像和相应文本）的标注数据集，使预训练的视觉语言模型（VLM）适应特定任务。这种方法可增强模型执行特定领域或特定任务功能的能力，如视觉问答（VQA）、图像文本描述生成、图表解析。

### **简介**

当你需要视觉语言模型（VLM）专门从事某个特定领域，或解决基础模型无法解决的问题时，有监督微调（SFT）就显得至关重要。例如，如果该模型在处理特殊的视觉特征或特定领域的术语方面表现不佳，有监督微调可让它从标注数据中学习，获得处理这些领域的任务的能力。

While SFT is highly effective, it has notable limitations:
虽然 SFT 非常高效，但它也有很多明显的局限性：
- **数据依赖**：针对特定任务的高质量标注数据是必须要有的
- **计算资源**：对大型的 VLM 进行微调需耗费大量计算资源
- **过拟合的风险**：如果微调范围过窄，模型可能会失去其泛化能力

尽管存在这些挑战，有监督微调（SFT）仍是一种强大的技术，可在特定场景中提升模型性能。

### **实践方法**

1. **数据准备**：从一个将图像与文本（如问题和答案）进行配对的标注数据集入手。例如，在图表分析这类任务中，“HuggingFaceM4/ChartQA” 数据集包含图表图像、查询内容以及简洁的回答。
2. **模型设置**：加载适合该任务的预训练视觉语言模型（VLM），比如 `HuggingFaceTB/SmolVLM - Instruct`，以及用于处理文本和图像输入的模型。针对监督学习并结合硬件条件对模型进行配置。
3. **微调过程**：
  - **数据格式化**：将数据集构建成类似聊天机器人的格式，把系统消息、用户查询以及相应答案进行配对。
  - **训练配置**：使用诸如 Hugging Face 的 `TrainingArguments` 或 TRL 的 `SFTConfig` 等工具来设置训练参数。这些参数包括 batch size、学习率和梯度累积步数，以优化资源使用。
  - **优化技术**：在训练过程中使用 **gradient checkpointing** 来节省内存。采用量化模型降低内存需求并加快计算速度。
  - 利用 TRL 库中的 `SFTTrainer` 训练器来简化训练流程。

## 偏好优化

偏好优化，尤其是直接偏好优化（DPO），旨在训练视觉语言模型（VLM）使其与人类偏好保持一致。该模型并非严格遵循预先设定的指令，而是学习对人类主观上更青睐的输出结果作为优先选择。这种方法在涉及创造性判断、精细化推理或存在多种可接受答案的任务中尤为实用。

### **简介**

偏好优化适用于那些人类主观偏好对任务成功起着关键作用的场景。通过在反映人类偏好的数据集上进行微调，直接偏好优化（DPO）提升了模型生成在上下文和风格上都符合用户期望的回复的能力。这种方法在创意写作、客户互动或多项选择等场景的任务中特别有效。

尽管偏好优化有诸多益处，但它也面临一些挑战：

- **数据质量**：需要高质量且带有偏好注释的数据集，这往往使数据收集成为瓶颈。
- **复杂性**：训练可能涉及复杂的过程，比如对偏好进行成对采样以及平衡计算资源。

偏好数据集必须清晰体现候选输出之间的偏好差异。例如，一个数据集可能会将一个问题与两个回复配对，一个是受偏好的，另一个则不太能被接受。模型会学习预测出那个受偏好的回复，即便它并非完全正确，只要它更符合人类的判断即可。

### **实践方法**
1. **数据准备**  
   准备好一个带有偏好注释的数据集对训练至关重要。每个数据样本一般包含一个问题和一副图片，以及两个候选回答：一个是受偏好的，另一个则不太能被接受。比如：
   - **问题**: How many families?  
     - **不倾向的回答**: The image does not provide any information about families.  
     - **倾向的回答**: The image shows a Union Organization table setup with 18,000 families.  

   该数据集引导模型优先选择偏好的回答，即便这些回答并非尽善尽美。

2. **模型配置**  
   加载一个预训练的视觉语言模型，准备好处理文本和图像输入的模型，结合 Hugging Face 的 `TRL` 库（支持 DPO 算法）写训练程序。针对监督学习进行模型配置，并使其适配你的硬件。

3. **训练流程** 
   训练需要配置 DPO 相关参数，主要有这些方向需要注意：

   - **数据集格式**：对每个样本进行结构化，整理好问题、图片、候选回答。
   - **损失函数**：需要使用和偏向性相关的函数，以此训练模型去选择倾向的回答。
   - **高效训练**：结合量化、梯度累积、LoRA 等技术，减少显存使用和计算量。

## 学习资源

- [Hugging Face Learn: Supervised Fine-Tuning VLMs](https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl) 
- [Hugging Face Learn: Supervised Fine-Tuning SmolVLM](https://huggingface.co/learn/cookbook/fine_tuning_smol_vlm_sft_trl)  
- [Hugging Face Learn: Preference Optimization Fine-Tuning SmolVLM](https://huggingface.co/learn/cookbook/fine_tuning_vlm_dpo_smolvlm_instruct)  
- [Hugging Face Blog: Preference Optimization for VLMs](https://huggingface.co/blog/dpo_vlm)

## 接下来

⏩ 学习 [vlm_finetune_sample_cn.ipynb](./notebooks/vlm_finetune_sample.ipynb)，动手实现 VLM 的偏好对齐训练。