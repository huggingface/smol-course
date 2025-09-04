import argparse
import json
from random import choices, sample

import argilla as rg
from distilabel.distiset import Distiset

################################################################################
# Script Parameters
################################################################################

parser = argparse.ArgumentParser(
    description="Annotate exam questions dataset using Argilla."
)
parser.add_argument(
    "--argilla_api_key",
    type=str,
    default="argilla.apikey",
    help="API key for Argilla",
)
parser.add_argument(
    "--argilla_api_url",
    type=str,
    default="http://localhost:6900",
    help="API URL for Argilla",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="exam_questions",
    help="Path to the exam questions dataset",
)
parser.add_argument(
    "--dataset_config",
    type=str,
    default="default",
    help="Configuration of the dataset",
)
parser.add_argument(
    "--dataset_split",
    type=str,
    default="train",
    help="Split of the dataset to use",
)
parser.add_argument(
    "--output_dataset_name",
    type=str,
    default="exam_questions",
    help="Name of the output Argilla dataset",
)

args = parser.parse_args()

################################################################################
# Create Argilla dataset with the feedback task for validation
################################################################################

client = rg.Argilla(api_key=args.argilla_api_key, api_url=args.argilla_api_url)

if client.datasets(args.output_dataset_name):
    print(f"Deleting existing dataset '{args.output_dataset_name}'")
    client.datasets(args.output_dataset_name).delete()

settings = rg.Settings(
    fields=[
        rg.TextField("question"),
        rg.TextField("answer_a"),
        rg.TextField("answer_b"),
        rg.TextField("answer_c"),
        rg.TextField("answer_d"),
    ],
    questions=[
        rg.LabelQuestion(
            name="correct_answer",
            labels=["answer_a", "answer_b", "answer_c", "answer_d"],
        ),
        rg.TextQuestion(
            name="improved_question",
            description="Could you improve the question?",
        ),
        rg.TextQuestion(
            name="improved_answer",
            description="Could you improve the best answer?",
        ),
    ],
)

dataset = rg.Dataset(settings=settings, name=args.output_dataset_name)
dataset.create()

################################################################################
# Load the Distiset and process and add records to Argilla dataset
# We will validate that questions appear in random order to avoid bias
# but we will show correct answers in the Argilla UI as suggestions.
################################################################################

distiset = Distiset.load_from_disk(args.dataset_path)
answer_names = ["answer_a", "answer_b", "answer_c", "answer_d"]
dataset_records = []

for exam in distiset[args.dataset_config][args.dataset_split]:
    exam_json = json.loads(exam["generation"])["exam"]

    for question in exam_json:
        answer = question["answer"]
        distractors = question["distractors"]
        distractors = choices(distractors, k=3)
        answers = distractors + [answer]
        answers = sample(answers, len(answers))
        suggestion_idx = answers.index(answer)
        fields = dict(zip(answer_names, answers))
        fields["question"] = question["question"]

        record = rg.Record(
            fields=fields,
            suggestions=[
                rg.Suggestion(
                    question_name="correct_answer",
                    value=answer_names[suggestion_idx],
                )
            ],
        )
        dataset_records.append(record)

dataset.records.log(dataset_records)

print(
    f"Dataset '{args.output_dataset_name}' has been created and populated in Argilla."
)
