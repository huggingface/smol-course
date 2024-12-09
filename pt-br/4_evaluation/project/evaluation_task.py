import argparse
import numpy as np

from lighteval.tasks.task import LightevalTaskConfig
from lighteval.tasks.task_utils import Doc
from lighteval.metrics.metrics import SampleLevelMetric, MetricCategory, MetricUseCase

################################################################################
# Script Parameters
################################################################################

parser = argparse.ArgumentParser(
    description="Define a LightEval task for exam questions."
)
parser.add_argument(
    "--hf_repo",
    type=str,
    default="burtenshaw/exam_questions",
    help="Hugging Face dataset repository ID",
)
parser.add_argument(
    "--hf_subset",
    type=str,
    default="default",
    help="Subset of the dataset to use",
)
parser.add_argument(
    "--evaluation_split",
    type=str,
    default="train",
    help="Split of the dataset to use for evaluation",
)

args = parser.parse_args()

################################################################################
# Define the prompt function based on the structure of the dataset
################################################################################


def prompt_fn(line, task_name: str = None):
    """Converts a dataset line to a Doc object for evaluation."""
    return Doc(
        task_name=task_name,
        query=line["question"],
        choices=[
            f" {line['answer_a']}",
            f" {line['answer_b']}",
            f" {line['answer_c']}",
            f" {line['answer_d']}",
        ],
        gold_index=["answer_a", "answer_b", "answer_c", "answer_d"].index(
            line["correct_answer"]
        ),
        instruction="Choose the correct answer for the following exam question:",
    )


################################################################################
# Define the custom metric based on guide here https://github.com/huggingface/lighteval/wiki/Adding-a-New-Metric
# Or use an existing metric based on the guide here: https://github.com/huggingface/lighteval/wiki/Metric-List
# Existing metrics can be imported from lighteval.metrics.metrics
################################################################################

custom_metric = SampleLevelMetric(
    metric_name="exam_question_accuracy",
    higher_is_better=True,
    category=MetricCategory.CLASSIFICATION,
    use_case=MetricUseCase.SCORING,
    sample_level_fn=lambda x: float(x["prediction"] == x["gold"]),
    corpus_level_fn=np.mean,
)

################################################################################
# Define the task based on the prompt function and the custom metric
# Based on the guide here: https://github.com/huggingface/lighteval/wiki/Adding-a-Custom-Task
################################################################################

task = LightevalTaskConfig(
    name="exam_questions",
    prompt_function=prompt_fn,
    suite=["community"],
    hf_repo=args.hf_repo,
    hf_subset=args.hf_subset,
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    metric=[custom_metric],
)

# Add the task to TASKS_TABLE
TASKS_TABLE = [task]

# MODULE LOGIC
if __name__ == "__main__":
    print([t.name for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
