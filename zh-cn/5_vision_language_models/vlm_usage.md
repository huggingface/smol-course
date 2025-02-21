# 视觉语言模型

视觉语言模型（VLMs）弥合了图像与文本之间的鸿沟，使得诸如生成图像的文字描述、基于视觉内容回答问题，以及理解文本与视觉数据之间的关系等高级任务得以实现。其架构旨在无缝处理这两种模态的数据。

### 模型架构

视觉语言模型将图像处理组件与文本生成模型相结合，以实现统一的理解。其架构的主要组成部分包括：

![VLM Architecture](./images/VLM_Architecture.png)

- **图像编码器（Image Encoder）**：用来将图片转化为用数值表示的特征，通常使用经训练过的 CLIP 或 ViT。
- **特征映射模块（Embedding Projector）**：用来将图片特征映射到文本特征空间，使其与文本特征兼容，通常使用全联接层或线性变换实现。
- **文本解码器（Text Decoder）**：作为文本生成模块，将融合的多模态信息转换成连贯的文字。举例来说，这部分可以用 Llama、Vicuna 等生成模型实现。
- **多模态映射模块（Multimodal Projector）**：: 提供了一个额外的层，来融合图像与文本的表征。对于像 LLaVA 这样的模型而言，这一层对于在两种模态之间建立更紧密的联系至关重要。

大多数视觉语言模型（VLMs）会利用预训练的图像编码器和文本解码器，并通过在图像-文本这样的成对数据上进行微调，使二者的特征相互对齐。这种方法既能提高训练效率，又能让模型有效地实现泛化。

### 使用方法

![VLM Process](./images/VLM_Process.png)

视觉语言模型（VLMs）被应用于一系列多模态任务。凭借其适应性，只需进行不同程度的微调，就能在多个领域发挥作用：

- **图像字幕生成**：为图像生成描述性文字。
- **视觉问答（VQA）**：针对图像内容回答相关问题。
- **跨模态检索**：为给定图像找到对应的文本，或反之，为给定文本找到对应的图像。
- **创意应用**：辅助设计、艺术创作，或生成引人入胜的多媒体内容。



![VLM Usage](./images/VLM_Usage.png)

训练和微调视觉语言模型（VLMs）依赖于高质量的数据集，这些数据集将图像与文本注释进行配对。像 Hugging Face 的 `transformers` 库这样的工具，为获取预训练的视觉语言模型提供了便捷途径，同时也简化了自定义微调的工作流程。

### Chat Format

很多视觉语言模型也是通过像聊天机器人一样的方式与用户互动的。这种格式包含以下信息：



- **系统信息（system message）**：为模型设定角色、上下文，例如“你是一个小助理，将会帮我分析视觉数据”这样。
- **用户查询（User queries）**：用户询问模型时输入给模型的信息，这包括视觉信息和文本信息
- **模型回答（Assistant responses）**：模型给出的文本形式的回复

这种对话结构直观易懂，符合用户预期，尤其适用于诸如客服或教育工具这类交互式应用程序。

下面就是一个格式化的输入示例：

```json
[
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a Vision Language Model specialized in interpreting visual data from chart images..."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "<image_data>"},
            {"type": "text", "text": "What is the highest value in the bar chart?"}
        ]
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "42"}]
    }
]
```

**多图输入或视频输入**

视觉语言模型（VLMs）还能够处理多张图像，甚至是视频。方法是调整输入结构，以适应连续或并行的视觉输入。对于视频，可以提取多帧并将其作为图像进行处理，同时保持时间顺序。

## 学习资源

- [Hugging Face Blog: Vision Language Models](https://huggingface.co/blog/vlms)
- [Hugging Face Blog: SmolVLM](https://huggingface.co/blog/smolvlm) 

## 接下来

⏩ 你可以尝试 [vlm_usage_sample_cn.ipynb](./notebooks/vlm_usage_sample.ipynb)，了解如何使用 SmolVLM 来处理各种任务。