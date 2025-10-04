import numpy as np

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.metrics.utils.metric_utils import (
    SampleLevelMetric,
    MetricCategory,
    MetricUseCase,
)

################################################################################
# データセットの構造に基づいてプロンプト関数を定義
################################################################################


def prompt_fn(line, task_name: str = None):
    """データセットの行を評価用のDocオブジェクトに変換します。"""
    instruction = "次の試験問題に対して正しい答えを選んでください："
    return Doc(
        task_name=task_name,
        query=f"{instruction} {line['question']}",
        choices=[
            f" {line['answer_a']}",
            f" {line['answer_b']}",
            f" {line['answer_c']}",
            f" {line['answer_d']}",
        ],
        gold_index=["answer_a", "answer_b", "answer_c", "answer_d"].index(
            line["correct_answer"]
        ),
        instruction=instruction,
    )


################################################################################
# カスタムメトリックを定義
# ガイドに基づいて https://github.com/huggingface/lighteval/wiki/Adding-a-New-Metric
# または既存のメトリックを使用 https://github.com/huggingface/lighteval/wiki/Metric-List
# 既存のメトリックは lighteval.metrics.metrics からインポート可能
################################################################################


def sample_level_fn(formatted_doc: Doc, **kwargs) -> bool:
    response = np.argmin(kwargs["choices_logprob"])
    return response == formatted_doc.gold_index


custom_metric = SampleLevelMetric(
    metric_name="exam_question_accuracy",
    higher_is_better=True,
    category=MetricCategory.MULTICHOICE,
    use_case=MetricUseCase.NONE,
    sample_level_fn=sample_level_fn,
    corpus_level_fn=np.mean,
)

################################################################################
# プロンプト関数とカスタムメトリックに基づいてタスクを定義
# ガイドに基づいて https://github.com/huggingface/lighteval/wiki/Adding-a-Custom-Task
################################################################################

task = LightevalTaskConfig(
    name="example",
    prompt_function=prompt_fn,
    suite=["community"],
    hf_repo="burtenshaw/exam_questions",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    metric=[custom_metric],
)

# TASKS_TABLEにタスクを追加
TASKS_TABLE = [task]

# モジュールロジック
if __name__ == "__main__":
    print([t.name for t in TASKS_TABLE])
    print(len(TASKS_TABLE))

# lighteval accelerate \
# "pretrained=HuggingFaceTB/SmolLM2-135M-Instruct" \
# "community|example|0|0" \
# --custom-tasks "submitted_tasks/example.py" \
# --output-dir "results"
