# 合成偏好数据集

在[前面章节](../2_preference_alignment/README_CN.md)，我们学习了直接偏好优化（DPO）。这里我们将学习如何为 DPO 之类的偏好优化算法生成训练数据集。我们的方法将基于前面[合成指令微调数据集](./instruction_datasets_cn.md)部分。此外，我们将展示如何通过基本提示或使用 EvolQuality 来添加额外的回答，以提高回答质量。最后，我们将展示如何使用 UltraFeedback 生成评分和评论。

## 生成多个回答

偏好数据集需要针对一个问题（`instruction`）的多个回答（`completions`）。通过提示语让模型生成两个回答当然是可行的，但我们要确保第二个回答不要和第一个过于相似（从质量和词汇上说）。这对训练非常重要，因为模型需要通过明显的差异来区分人类偏好。这里我们还需要判断哪个回答是我们倾向的（`chosen`）、哪个是我们不倾向的（`rejected`）。在[生成分数](#生成分数)部分我们会讲解。 

### Model pooling

Model pooling 是一个很简单的方法：你可以使用不同的模型针对同意问题生成不同的回答。如果想进一步改进第二个回答的质量，你还可以调节不同的生成参数，如 `temperature`。通过不同的提示语模板或系统提示语，你也可以生成有多样性的回答。理论上讲，用两个不同质量的模型生成两个回答，选择质量好的那个回答就可以完成这件事情。

这里我们通过 [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) 和 [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) 这两个模型进行 model pooling。针对每一个问题，我们都可以生成两个回答。同样地，我们首先用 `LoadDataFromDicts` 载入种子数据，然后用 `>>` 运算符串联 pipeline 的数据流，用 `[]` 取结果。

```python
from distilabel.llms import TransformersLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import GroupColumns, LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration

with Pipeline() as pipeline:
    data = LoadDataFromDicts(data=[{"instruction": "What is synthetic data?"}])
    llm_a = TransformersLLM(model="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    gen_a = TextGeneration(llm=llm_a)
    llm_b = TransformersLLM(model="Qwen/Qwen2.5-1.5B-Instruct")
    gen_b = TextGeneration(llm=llm_b)
    group = GroupColumns(columns=["generation"])
    data >> [gen_a, gen_b] >> group

if __name__ == "__main__":
    distiset = pipeline.run()
    print(distiset["default"]["train"]["grouped_generation"][0])
# {[
#   'Synthetic data is artificially generated data that mimics real-world usage.',
#   'Synthetic data refers to data that has been generated artificially.'
# ]}
```

可以看到，我们这里获取到了两个回答。如果想提升回答的多样性，还可以在 `TextGeneration` 这一步提供 `system_prompt` 或给 `TransformersLLM` 传递生成参数。接下来我们看看怎样用 EvolQuality 进一步提升回答的质量。

### EvolQuality

EvolQuality 和 [EvolInstruct](./instruction_datasets_cn.md#evolinstruct) 类似，都是提示语技术。但 EvolQuality 改进的是回答的质量，而不是输入的问题。EvolQuality 会把问题和回答都作为输入，然后根据提供的标准不断修改进化回答。根据有帮助度、相关性、深度、创造性和其它细节，我们定义一个更好的回答。这个更好的回答就可以加入数据集。理论上，我们可以认为这个“更好的回答”就是我们偏好优化时倾向的回答。

具体的提示语[在 distilabel 已有实现](https://github.com/argilla-io/distilabel/tree/main/src/distilabel/steps/tasks/evol_quality)，这里我们展示一个简化版本：

```bash
I want you act as a Response Rewriter.
Given prompt a and a response, rewrite the response into a better version.
Complicate the prompt based on the following criteria:
{{ criteria }}

# Prompt
{{ input }}

# Response
{{ output }}

# Improved Response
```

代码实现上，我们需要用 [EvolQuality](https://distilabel.argilla.io/dev/components-gallery/tasks/evolquality/) 这个类。这里代码只进化迭代一次。

```python
from distilabel.llms import TransformersLLM
from distilabel.steps.tasks import EvolQuality

llm = TransformersLLM(model="HuggingFaceTB/SmolLM2-1.7B-Instruct")
evol_quality = EvolQuality(llm=llm, num_evolutions=1)
evol_quality.load()

instruction = "What is synthetic data?"
completion = "Synthetic data is artificially generated data that mimics real-world usage."

next(evol_quality.process([{
    "instruction": instruction,
    "response": completion
}]))
# The process of generating synthetic data through manual prompting involves creating artificial data sets that mimic real-world usage patterns.
```

可以看到，回答与问题对应，且已经更复杂了。这是个好的开端，但如果你做过 `EvolInstruct` 的实验，就会知道这种算法得到的“更好”回答有时并没有变好。所以我们还需要加入额外的测评技术，确保数据集不出现错误标注。接下来部分就会讨论这一技术。

## 生成分数

分数是衡量一个回答相较于另一个回答受偏好程度的指标。一般来说，这些分数可以是绝对的、主观的或相对的。在本课程中，我们将重点关注前两种，因为它们对于创建偏好数据集最为有用。这种评分方式是一种利用语言模型进行判断和评估的方法，因此与我们在[评测](../4_evaluation/README_CN.md)这一章中看到的评估技术存在一些重叠。与其他评估技术一样，分数和评估通常需要更大的模型，以便更好地契合人类偏好。

### UltraFeedback

UltraFeedback 可以对给出的问题和回答生成分数和评价。

回答的分数一般基于一定的标准给出。有四类细化的标准：有用性、相关性、深度、创造性。借助这些标准，可以简化评分过程。得到分数后，我们就可以决定哪个回答是倾向的（`chosen`），哪个是不倾向的（`rejected`）。因为分数是绝对的，它也可以被用来过滤质量极低或区分度不高的数据。

此外，评价信息可以为得分提供一些逻辑道理，可以用来理解分数造成差异的原因。

UltraFeedback 对应的提示语在 [distilabel 中也有实现](https://github.com/argilla-io/distilabel/tree/main/src/distilabel/steps/tasks/templates/ultrafeedback)，这里提供简化版本：

```bash
Evaluate the model's outputs based on various criteria: Helpfulness, Relevance, Deepening, Creativity
Your role is to provide a holistic assessment based on the above factors.
Score the output from 1 to 5 on overall quality.

Answer with the following format: score - rationale

# Input
{{ input }}

# Response
{{ output }}

# Score - Rationale
```

在代码层面，我们使用 [UltraFeedback](https://distilabel.argilla.io/dev/components-gallery/tasks/ultrafeedback/) 类，评估取自 [Model Pooling](#model-pooling) 这一部分的合成的 `prompt` 和 `completion`。

```python
from distilabel.llms import TransformersLLM
from distilabel.steps.tasks import UltraFeedback

llm = TransformersLLM(model="HuggingFaceTB/SmolLM2-1.7B-Instruct")
ultrafeedback = UltraFeedback(llm=llm)
ultrafeedback.load()

instruction = "What is synthetic data?"
completion_a = "Synthetic data is artificially generated data that mimics real-world usage."
completion_b = "Synthetic data refers to data that has been generated artificially."

next(ultrafeedback.process([{
    "instruction": instruction,
    "generations": [completion_a, completion_b]
}]))
# [
#     {
#         'ratings': [4, 5],
#         'rationales': ['could have been more specific', 'good definition'],
#     }
# ]
```

## 最佳实践

- 相比于生成分数和评价，只生成分数一般会更简单、更少耗费资源
- 用更大的模型去生成分数和评价
- 建议使用一系列有多样性的模型，去生成分数和评价
- 不断优化 `system_prompt` 和模型

## 接下来

👨🏽‍💻 代码 - 通过[练习](./notebooks/preference_dpo_dataset.ipynb) 去生成一个偏好对齐的数据集

## 参考资料

- [Distilabel 官方文档](https://distilabel.argilla.io/latest/)
- [Deita 论文](https://arxiv.org/abs/2312.15685)
- [UltraFeedback 论文](https://arxiv.org/abs/2310.01377)
