# 自动化基准测试

自动化基准测试可以作为一个标准化的工具，来衡量语言模型在不同任务上的能力。不过，虽然它可以有效地用来了解模型当前的性能，但我们还要认识到，这些评测结果仅仅是模型全面评估的一小部分，不能完全反映模型性能。

## 理解自动化基准评测

自动化基准评测一般是在定义好的领域和评测指标下，对特定的数据集进行测试。这种基准测试会评估模型多方面的能力，从最基础的语言理解到复杂的逻辑推理。其最主要的优点还是标准型方面，不同模型都使用相同标准来测试，可以用来对比不同模型的效果，同时测评出的结果也是可复现的。

然而，我们也需要知道，这种基准测试也不是完全反映模型的现实能力的。比如，一个在学术评测基准上表现优异的模型，也许在其它应用领域或实践层面表现很差。

## 现有评测基准及其局限性

### 通识层面的评测基准

MMLU（Massive Multitask Language Understanding）这个评测基准，涵盖了从科学到人文共 57 个学科的知识，是一个通识层面的评测基准。但虽然全面，它可能在某些领域的专业深度还不算够。另一方面，TruthfulQA 这个评测基准，涵盖 38 个学科问答，则会评测模型输出的真实性如何。

### 推理层面的评测基准

BBH（Big Bench Hard）和 GSM8K 这两个评测基准重点关注复杂的推理任务。BBH 主要测试逻辑思考和规划能力，GSM8K 则特别关注数学问题的求解。这些评测基准可以用来评测模型的分析问题能力，但人在现实世界中微妙的推理细节可能会被忽略。

### 语言理解层面的评测基准

在语言理解层面，HELM 提供了一个全面的评测框架，而 WinoGrande 则通过代词指代的歧义消除，测试模型在常识层面的能力。这些评测基准让我们能深入了解模型在语言处理层面的水平，但缺点是暂未模仿到人与人之间对话的复杂性，同时暂未测评到专业术语。

## 其它评测方法

除了上述评基准，很多机构也开发了其它评测方法，以应对标准化基准测试的缺陷：

### 用大语言模型作为评审

用一个大语言模型去评测另一个大语言模型的输出，这种方法最近开始常用起来。相比于传统的评测指标，这种方法可以提供更细致入微的反馈。缺点是作为评审的大语言模型自己也有偏见和局限性，可能导致评测结果不够好。

### 在竞技场内相互评测

像 Anthropic's Constitutional AI Arena 这样的平台，可以让模型在里面相互互动、评测。这样的评测场景也有助于模型发现各自的强项和弱点。

### 自定义评测基准

很多组织也会自己开发对内的评测基准，通常是针对特定的需求或应用场景开发的。这样开发出来的评测基准一般包含特定的专业领域知识、反映产品的实际应哟过场景。

## Creating Your Own Evaluation Strategy

虽然使用 LightEval 可以很方便地进行标准化的基准评测，但作为一个 LLM 开发者，你必须也要针对你们产品的应用场景开发自己的评测方案。标准化的基准评测仅仅是一测评开始的第一步，你绝不能只用它进行模型测试。

如何自定义你的方位测评？方法如下：

1. 首先从相关的标准化基准测试中开始，建立一个基准，保证能够和其它模型进行对比。

2. 针对你的应用场景的独特需求，确认你的模型将会应对的挑战。例如，你的模型上线后将会主要执行什么任务？有可能出现哪些问题？哪些 bad case 是最应该避免的？

3. 开发你自己的测试数据集，以便专门应对你的测试场景。这可能包括：
   - 在你的特定领域里，真实用户的请求
   - 常见的边缘案例
   - 可能发生的有挑战性的情况

4. 也需要考虑开发一个多层级的评测策略：
   - 首先，为了能快速获取反馈，你可以设置一个自动化的评测指标
   - 针对细微的语言理解能力，考虑人工测评
   - 针对专业领域的应用，引入行业专家的评审
   - 在控制变量的环境下，进行 A/B 测试

## 使用 LightEval 进行基准评测

LightEval 评测任务通过以下格式定义：
```
{suite}|{task}|{num_few_shot}|{auto_reduce}
```

- **suite**：哪一套基准评测方法（比如 'mmlu'、'truthfulqa'）
- **task**：这一套基准评测方法中的哪个任务（比如 'abstract_algebra'）
- **num_few_shot**：提示词中加入的示例的数量（如果是 0，那就是 zero-shot 测试）
- **auto_reduce**：当提示词太长时，是否自动减少提示词中 few-shot 的样本量

举例来说，`"mmlu|abstract_algebra|0|0"` 就会评测 MMLU 的 abstract algebra 任务，推理是 zero-shot 形式。

### 评测代码示例

以下代码就是一个在某个领域进行自动化评测的示例：

```python
from lighteval.tasks import Task, Pipeline
from transformers import AutoModelForCausalLM

# Define tasks to evaluate
domain_tasks = [
    "mmlu|anatomy|0|0",
    "mmlu|high_school_biology|0|0", 
    "mmlu|high_school_chemistry|0|0",
    "mmlu|professional_medicine|0|0"
]

# Configure pipeline parameters
pipeline_params = {
    "max_samples": 40,  # Number of samples to evaluate
    "batch_size": 1,    # Batch size for inference
    "num_workers": 4    # Number of worker processes
}

# Create evaluation tracker
evaluation_tracker = EvaluationTracker(
    output_path="./results",
    save_generations=True
)

# Load model and create pipeline
model = AutoModelForCausalLM.from_pretrained("your-model-name")
pipeline = Pipeline(
    tasks=domain_tasks,
    pipeline_parameters=pipeline_params,
    evaluation_tracker=evaluation_tracker,
    model=model
)

# Run evaluation
pipeline.evaluate()

# Get and display results
results = pipeline.get_results()
pipeline.show_results()
```

测评结果会以表格形式打印出来：

```
|                  Task                  |Version|Metric|Value |   |Stderr|
|----------------------------------------|------:|------|-----:|---|-----:|
|all                                     |       |acc   |0.3333|±  |0.1169|
|leaderboard:mmlu:_average:5             |       |acc   |0.3400|±  |0.1121|
|leaderboard:mmlu:anatomy:5              |      0|acc   |0.4500|±  |0.1141|
|leaderboard:mmlu:high_school_biology:5  |      0|acc   |0.1500|±  |0.0819|
```

使用 pandas 的 `DataFrame` 或其它可视化方式呈现结果也是可以的。

# 接下来

⏩ 学习 [Custom Domain Evaluation](./custom_evaluation_cn.md)，了解如何根据你的特定需求创建自定义的评测流程。
