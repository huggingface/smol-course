# 在特定领域进行自定义评测

虽然标准化的评测基准让我们对模型的性能有了初步的认识，但针对特定的应用场景，我们还需要专门制定评测方法，考察模型在特定领域的表现。本文将带你创建自定义的评测流程，针对你的目标领域对模型进行精准评测。

## 设计评测策略

成功创建自定义评测策略的第一步是确定清晰的目标。你需要考虑在你的特定领域，哪些特殊能力是你的模型需要具备的？这可能涉及技术层面的知识、推理的模式、特定的格式等。你需要认真记录好这些需求，然后参考这些需求去设计测试任务、选择评测指标。

测试的样例不仅需要包含标准应用场景，也要考虑边缘场景。举例来说，如果是医学领域，常见的诊断场景和罕见情况都是需要考虑的。在金融领域，除了常规交易，复杂交易（比如设计多种货币或特殊条件的情况）的处理能力也需要被测试到。

## 使用 LightEval 的代码实现

LightEval 是一个非常灵活的框架，可以用来实现自定义的测评任务。下面代码展示了如何创建自定义测试任务：

```python
from lighteval.tasks import Task, Doc
from lighteval.metrics import SampleLevelMetric, MetricCategory, MetricUseCase

class CustomEvalTask(Task):
    def __init__(self):
        super().__init__(
            name="custom_task",
            version="0.0.1",
            metrics=["accuracy", "f1"],  # Your chosen metrics
            description="Description of your custom evaluation task"
        )
    
    def get_prompt(self, sample):
        # Format your input into a prompt
        return f"Question: {sample['question']}\nAnswer:"
    
    def process_response(self, response, ref):
        # Process model output and compare to reference
        return response.strip() == ref.strip()
```

## 自定义评价指标

特定领域的测试任务通常也需要特殊的评价指标。LightEval 也可以灵活地做到这一点：

```python
from aenum import extend_enum
from lighteval.metrics import Metrics, SampleLevelMetric, SampleLevelMetricGrouping
import numpy as np

# Define a sample-level metric function
def custom_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> dict:
    """Example metric that returns multiple scores per sample"""
    response = predictions[0]
    return {
        "accuracy": response == formatted_doc.choices[formatted_doc.gold_index],
        "length_match": len(response) == len(formatted_doc.reference)
    }

# Create a metric that returns multiple values per sample
custom_metric_group = SampleLevelMetricGrouping(
    metric_name=["accuracy", "length_match"],  # Names of sub-metrics
    higher_is_better={  # Whether higher values are better for each metric
        "accuracy": True,
        "length_match": True
    },
    category=MetricCategory.CUSTOM,
    use_case=MetricUseCase.SCORING,
    sample_level_fn=custom_metric,
    corpus_level_fn={  # How to aggregate each metric
        "accuracy": np.mean,
        "length_match": np.mean
    }
)

# Register the metric with LightEval
extend_enum(Metrics, "custom_metric_name", custom_metric_group)
```

如果每个样例只有一个指标，代码可以是这样：

```python
def simple_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> bool:
    """Example metric that returns a single score per sample"""
    response = predictions[0]
    return response == formatted_doc.choices[formatted_doc.gold_index]

simple_metric_obj = SampleLevelMetric(
    metric_name="simple_accuracy",
    higher_is_better=True,
    category=MetricCategory.CUSTOM,
    use_case=MetricUseCase.SCORING,
    sample_level_fn=simple_metric,
    corpus_level_fn=np.mean  # How to aggregate across samples
)

extend_enum(Metrics, "simple_metric", simple_metric_obj)
```

实现完代码后，你就可以在你的评测任务中引用这些指标的名称，然后在你的评测任务中使用。这些指标会在测试过程中自动在每个样本上计算，并最终统计数值。

如果需要使用更复杂的评测指标，你还可以实现这些功能：

- 使用元数据，对不同测试样本的分数进行加权或其它调整
- 对所有样本的指标进行统计时，你可以实现一个自定义的函数（上述示例中 corpus-level 统计使用了取平均的方法）
- 对输入到你的评测指标函数中的数据进行格式检查
- 记录边缘场景及其期望的行为



你可以学习本章 [domain evaluation project](./project/README_CN.md) 这个项目课程，真正地动手实践一下自定义测评。

## 测试数据集的创建

高质量的测评需要高质量的测试数据集。在创建数据集时，需要考虑以下方面：

1. 专家级的标注：与领域专家一起创建和检验测试样本。你可以用 [Argilla](https://github.com/argilla-io/argilla) 高效地进行标注。

2. 真实世界的数据：收集真实数据并进行脱敏，确保这些样本能代表真实部署模型的场景。

3. 借助合成数据：使用 LLM 生成一些初始样本，然后让领域专家检查、修改。这样可以助你快速创建数据集。

## 最佳实践

- 全面记录你的测试方法，包括各种假设和局限
- 保证测试样本的多样性，确保你的领域内各个方面都能被测试到
- 如有需要，自动化的测试指标和人工评测都要用上
- 对测评数据集和代码进行版本控制
- 定期更新你的评测流程，不断加入新的边缘场景、完善新的需求

## 参考资料

- [LightEval 如何添加自定义任务](https://github.com/huggingface/lighteval/wiki/Adding-a-Custom-Task)
- [LightEval 如何添加自定义测评指标](https://github.com/huggingface/lighteval/wiki/Adding-a-New-Metric)
- [Argilla 文档](https://docs.argilla.io) 可以用来进行数据标注
- [评测的指南书籍](https://github.com/huggingface/evaluation-guidebook) 大语言模型评测领域的全面指南
- 
# 接下来

⏩ 完整的自定义测评请见本章 [domain evaluation project](./project/README_CN.md)。