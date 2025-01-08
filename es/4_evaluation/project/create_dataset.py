import argparse

import argilla as rg
from datasets import Dataset

################################################################################
# Script Parameters
################################################################################

parser = argparse.ArgumentParser(
    description="Create a Hugging Face dataset from annotated Argilla data."
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
    help="Path to the Argilla dataset",
)
parser.add_argument(
    "--dataset_repo_id",
    type=str,
    default="burtenshaw/exam_questions",
    help="Hugging Face dataset repository ID",
)

args = parser.parse_args()

################################################################################
# Initialize Argilla client and load dataset
################################################################################

client = rg.Argilla(api_key=args.argilla_api_key, api_url=args.argilla_api_url)
dataset = client.datasets(args.dataset_path)

################################################################################
# Process Argilla records
################################################################################

dataset_rows = []

for record in dataset.records(with_suggestions=True, with_responses=True):
    row = record.fields

    if len(record.responses) == 0:
        answer = record.suggestions["correct_answer"].value
        row["correct_answer"] = answer
    else:
        for response in record.responses:
            if response.question_name == "correct_answer":
                row["correct_answer"] = response.value
    dataset_rows.append(row)

################################################################################
# Create Hugging Face dataset and push to Hub
################################################################################

hf_dataset = Dataset.from_list(dataset_rows)
hf_dataset.push_to_hub(repo_id=args.dataset_repo_id)

print(f"Dataset has been successfully pushed to {args.dataset_repo_id}")
