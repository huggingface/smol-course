# smol-course 中文翻译技术术语表 (Technical Glossary)

本文档定义了 smol-course 中文翻译中使用的技术术语标准翻译。请在翻译过程中保持一致性。

## 翻译原则

1. **保留英文缩写**：DPO, LoRA, PEFT, SFT, RLHF 等保持英文
2. **混合格式**：首次出现时使用"中文翻译 (English Abbreviation)"格式
3. **后续提及**：可以只使用中文或只使用英文缩写
4. **代码相关**：所有代码、函数名、库名保持英文不变

---

## 核心概念 (Core Concepts)

| English | 中文 | 首次使用格式 | 备注 |
|---------|------|------------|------|
| Fine-tuning | 微调 | 微调 (Fine-tuning) | 通用术语 |
| Instruction tuning | 指令微调 | 指令微调 (Instruction Tuning) | |
| Supervised Fine-Tuning (SFT) | 监督微调 | 监督微调 (SFT) | 保留 SFT 缩写 |
| Preference alignment | 偏好对齐 | 偏好对齐 (Preference Alignment) | |
| Chat template | 对话模板 | 对话模板 (Chat Template) | |
| Base model | 基础模型 | 基础模型 | |
| Foundation model | 基础模型 | 基础模型 | |
| Language model | 语言模型 | 语言模型 | |
| Large Language Model (LLM) | 大语言模型 | 大语言模型 (LLM) | 保留 LLM 缩写 |

## 训练技术 (Training Techniques)

| English | 中文 | 首次使用格式 | 备注 |
|---------|------|------------|------|
| Direct Preference Optimization (DPO) | 直接偏好优化 | 直接偏好优化 (DPO) | 保留 DPO 缩写 |
| RLHF | 人类反馈强化学习 | 人类反馈强化学习 (RLHF) | Reinforcement Learning from Human Feedback |
| ORPO | 比值偏好优化 | 比值偏好优化 (ORPO) | Odds Ratio Preference Optimization |
| LoRA | 低秩适应 | 低秩适应 (LoRA) | Low-Rank Adaptation |
| PEFT | 参数高效微调 | 参数高效微调 (PEFT) | Parameter-Efficient Fine-Tuning |
| Prompt tuning | 提示词微调 | 提示词微调 (Prompt Tuning) | |
| PPO | 近端策略优化 | 近端策略优化 (PPO) | Proximal Policy Optimization |
| Reinforcement Learning | 强化学习 | 强化学习 | |
| Gradient descent | 梯度下降 | 梯度下降 | |

## 模型组件 (Model Components)

| English | 中文 | 首次使用格式 | 备注 |
|---------|------|------------|------|
| Token | 词元 | 词元 (Token) | |
| Tokenizer | 分词器 | 分词器 (Tokenizer) | |
| Embedding | 嵌入 | 嵌入 (Embedding) | |
| Layer | 层 | 层 | |
| Attention | 注意力机制 | 注意力机制 (Attention) | |
| Attention head | 注意力头 | 注意力头 | |
| Transformer | Transformer | Transformer | 保持英文 |
| Encoder | 编码器 | 编码器 | |
| Decoder | 解码器 | 解码器 | |
| Hidden state | 隐藏状态 | 隐藏状态 | |
| Checkpoint | 检查点 | 检查点 (Checkpoint) | |
| Weight | 权重 | 权重 | |
| Parameter | 参数 | 参数 | |
| Bias | 偏置 | 偏置 | |

## 数据集 (Datasets)

| English | 中文 | 首次使用格式 | 备注 |
|---------|------|------------|------|
| Dataset | 数据集 | 数据集 | |
| Synthetic dataset | 合成数据集 | 合成数据集 | |
| Training set | 训练集 | 训练集 | |
| Validation set | 验证集 | 验证集 | |
| Test set | 测试集 | 测试集 | |
| Batch | 批次 | 批次 (Batch) | |
| Prompt | 提示词 | 提示词 (Prompt) | |
| Response | 响应/回复 | 响应 | 根据上下文选择 |
| Preference pair | 偏好对 | 偏好对 | |
| Chosen/Preferred | 优选/首选 | 优选响应 | |
| Rejected | 拒绝/非优选 | 拒绝响应 | |

## 评估 (Evaluation)

| English | 中文 | 首次使用格式 | 备注 |
|---------|------|------------|------|
| Benchmark | 基准测试 | 基准测试 (Benchmark) | |
| Metric | 指标 | 指标 (Metric) | |
| Accuracy | 准确率 | 准确率 | |
| Precision | 精确率 | 精确率 | |
| Recall | 召回率 | 召回率 | |
| F1 Score | F1 分数 | F1 分数 | |
| Loss | 损失 | 损失 (Loss) | |
| Loss function | 损失函数 | 损失函数 | |
| Perplexity | 困惑度 | 困惑度 (Perplexity) | |
| Evaluation | 评估 | 评估 | |

## 视觉语言模型 (Vision Language Models)

| English | 中文 | 首次使用格式 | 备注 |
|---------|------|------------|------|
| Vision Language Model (VLM) | 视觉语言模型 | 视觉语言模型 (VLM) | |
| Multimodal | 多模态 | 多模态 (Multimodal) | |
| Image encoder | 图像编码器 | 图像编码器 | |
| Vision encoder | 视觉编码器 | 视觉编码器 | |
| Image-text pair | 图文对 | 图文对 | |

## 训练参数 (Training Parameters)

| English | 中文 | 首次使用格式 | 备注 |
|---------|------|------------|------|
| Learning rate | 学习率 | 学习率 (Learning Rate) | |
| Epoch | 轮次/周期 | 训练轮次 (Epoch) | |
| Batch size | 批次大小 | 批次大小 (Batch Size) | |
| Gradient accumulation | 梯度累积 | 梯度累积 | |
| Gradient clipping | 梯度裁剪 | 梯度裁剪 | |
| Optimizer | 优化器 | 优化器 (Optimizer) | |
| Scheduler | 调度器 | 学习率调度器 | |
| Warmup | 预热 | 预热 (Warmup) | |
| Overfitting | 过拟合 | 过拟合 (Overfitting) | |
| Underfitting | 欠拟合 | 欠拟合 (Underfitting) | |
| Hyperparameter | 超参数 | 超参数 (Hyperparameter) | |

## 编程和工具 (Programming & Tools)

| English | 中文 | 首次使用格式 | 备注 |
|---------|------|------------|------|
| Notebook | 笔记本 | Jupyter 笔记本 | |
| Google Colab | Google Colab | Google Colab | 品牌名，保持英文 |
| Hugging Face | Hugging Face | Hugging Face | 品牌名，保持英文 |
| Repository | 代码仓库 | 代码仓库 (Repository) | |
| Pull request (PR) | 拉取请求 | 拉取请求 (Pull Request/PR) | |
| Commit | 提交 | 提交 (Commit) | |
| Pipeline | 流程/管道 | 推理流程 (Pipeline) | 根据上下文 |
| Inference | 推理 | 推理 (Inference) | |
| Deployment | 部署 | 部署 (Deployment) | |

## 库和框架名称 (Libraries & Frameworks)

**重要**：以下库和框架名称始终保持英文，不翻译

- `transformers`
- `datasets`
- `trl` (Transformers Reinforcement Learning)
- `peft`
- `accelerate`
- `torch` / `PyTorch`
- `TensorFlow`
- `numpy`
- `pandas`

## 模型名称 (Model Names)

**重要**：模型名称保持英文，不翻译

- SmolLM / SmolLM2 / SmolLM3
- GPT / GPT-2 / GPT-3 / GPT-4
- BERT
- Llama / Llama 2 / Llama 3
- Mistral
- Gemma

## 特殊标记和语法 (Special Tokens & Syntax)

**重要**：以下内容保持原样，不翻译

- `<|im_start|>`, `<|im_end|>` - 特殊标记
- `<CourseFloatingBanner>` - JSX 组件
- 所有 HTML/JSX 标签
- 文件路径和 URL
- 代码块中的所有代码
- 变量名、函数名、类名

## 常用短语翻译

| English | 中文 |
|---------|------|
| Best practices | 最佳实践 |
| Step by step | 逐步 |
| Hands-on | 实践/动手实践 |
| Tutorial | 教程 |
| Guide | 指南 |
| Example | 示例 |
| Implementation | 实现 |
| Configuration | 配置 |
| Setup | 设置/配置 |
| Installation | 安装 |
| Prerequisites | 前置要求/先决条件 |
| Requirements | 要求/依赖 |

## 翻译示例

### 示例 1：首次提及
英文：Direct Preference Optimization (DPO) provides a simpler approach to preference alignment.

中文：直接偏好优化 (DPO) 为偏好对齐提供了一种更简单的方法。

### 示例 2：后续提及
英文：DPO is more efficient than RLHF.

中文：DPO 比 RLHF 更高效。
或：直接偏好优化比人类反馈强化学习更高效。

### 示例 3：代码块
```python
# 英文注释：Load model and tokenizer
# 中文翻译：加载模型和分词器
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("SmolLM3-3B")
```

---

## 更新记录

- 2024-01: 初始版本创建
