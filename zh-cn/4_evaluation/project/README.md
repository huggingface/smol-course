# 用 Argilla、Distilabel 和 LightEval 进行特定领域评测

绝大多数的常用评测基准关注的都是模型非常基本的能力，比如推理、数学、编程等，而忽略了特定的专业领域能力。如何进行专业领域（如金融、法律、医学等）的模型评测呢？

本教程将会展示**自定义领域**模型测试的完整流程。我们重点关注数据部分。在相关数据收集、标注测试数据方面，我们会使用 [Argilla](https://github.com/argilla-io/argilla)、[Distilabel](https://github.com/argilla-io/distilabel) 和 [LightEval](https://github.com/huggingface/lighteval) 作为工具，生成考试问题相关的数据。


## 项目结构

本项目包含四份 Python 代码文件。我们分四个步骤完成模型在自定义领域的测评，每份代码对应一个步骤。这四个步骤分别是：数据生成、数据标注、相关测试样本的提取，以及模型评测。

| 代码文件 | 概述 |
|-------------|-------------|
| generate_dataset.py | 使用一个专门的语言模型，从多个文本文档中生成考试问题。 |
| annotate_dataset.py | 用 Argilla 创建一个数据集，手动为生成的考试问题数据进行标注。 |
| create_dataset.py | 处理标注过的数据，并创建对应的 HuggingFace 数据集。 |
| evaluation_task.py | 自定义了一个 LightEval 任务，在前面建立好的测试数据上测试。 |

## 步骤

### 1. 数据集的生成

使用 `generate_dataset.py`，我们可以用 `distilabel` 这个库根据几个文本文档生成一些考试问题。一个特定的模型（这里默认使用 Meta-Llama-3.1-8B-Instruct）被拿来用以生成问题、正确答案和错误答案（错误答案用来作为干扰项）。你需要加入你自己的数据，也可以切换使用别的模型。

通过以下命令可以开始生成：

```sh
python generate_dataset.py --input_dir path/to/your/documents --model_id your_model_id --output_path output_directory
```

代码运行中将会创建一个 [Distiset](https://distilabel.argilla.io/dev/sections/how_to_guides/advanced/distiset/)，它包含根据文档生成的考试问题，其中保存文档的目录是 `input_dir`。

### 2. 标注数据集

使用 `annotate_dataset.py` 可以将生成的问题创建为一个 Argilla 数据集，用以标注。该程序会构建起数据集结构，并把生成的问题和回答填充进统一结构中，还可以改变数据样本的顺序来避免偏向性。使用 Argilla，你或者一个领域专家可以用选择的方式给出每个问题的正确回答。

在标注界面中，已生成的几个回答会随机排列着，供标注人员选择正确答案。这其中，LLM 会提供一个建议的正确回答，你可以选择你认为正确的回答，也可以对选中的正确回答进行编辑改进。标注过程的耗时取决于你的数据集规模、领域内数据的复杂度，以及 LLM 的能力强弱。举例来说，我们是可以借助 Llama-3.1-70B-Instruct 在 1 小时内标注好迁移学习领域 150 个样本的，大多数时候，直接选择正确答案即可。

通过以下命令可以开始标注：


```sh
python annotate_dataset.py --dataset_path path/to/distiset --output_dataset_name argilla_dataset_name
```

这将会创建一个 Argilla 数据集，用来手工检查和标注数据。

![argilla_dataset](./images/domain_eval_argilla_view.png)

可以参考[这里的指引](https://docs.argilla.io/latest/getting_started/quickstart/)，在本地或 Hugging Face 的 space 里部署标注任务。

### 3. 创建数据集

使用 `create_dataset.py` 可以进一步处理 Argilla 标注的数据，并创建一个 Hugging Face 数据集。这个数据集里每条数据包含这些内容：问题、可能的回答、正确回答（所在的列的名字）。运行以下命令即可创建最终的数据集：

```sh
huggingface_hub login
python create_dataset.py --dataset_path argilla_dataset_name --dataset_repo_id your_hf_repo_id
```

最终，数据集会被推送到 Hugging Face Hub 里。本示例的数据集已经上传了，可以在[这里](https://huggingface.co/datasets/burtenshaw/exam_questions/viewer/default/train)查看，大致是这样：

![hf_dataset](./images/domain_eval_dataset_viewer.png)

### 4. 开始测评

使用 `evaluation_task.py`，你可以自定义一个 LightEval 任务，用来在前面创建的数据集上测试模型。具体执行命令如下：

```sh
lighteval accelerate \
    --model_args "pretrained=HuggingFaceH4/zephyr-7b-beta" \
    --tasks "community|exam_questions|0|0" \
    --custom_tasks domain-eval/evaluation_task.py \
    --output_dir "./evals"
```

此外，lighteval 的 wiki 也提供了更详细的讲解：

- [自定义评测任务](https://github.com/huggingface/lighteval/wiki/Adding-a-Custom-Task)
- [自定义评测指标](https://github.com/huggingface/lighteval/wiki/Adding-a-New-Metric)
- [使用已有的评测指标](https://github.com/huggingface/lighteval/wiki/Metric-List)
