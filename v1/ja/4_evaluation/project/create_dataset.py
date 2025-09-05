import argparse

import argilla as rg
from datasets import Dataset

################################################################################
# スクリプトのパラメータ
################################################################################

parser = argparse.ArgumentParser(
    description="注釈付きArgillaデータからHugging Faceデータセットを作成します。"
)
parser.add_argument(
    "--argilla_api_key",
    type=str,
    default="argilla.apikey",
    help="ArgillaのAPIキー",
)
parser.add_argument(
    "--argilla_api_url",
    type=str,
    default="http://localhost:6900",
    help="ArgillaのAPI URL",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="exam_questions",
    help="Argillaデータセットのパス",
)
parser.add_argument(
    "--dataset_repo_id",
    type=str,
    default="burtenshaw/exam_questions",
    help="Hugging FaceデータセットリポジトリID",
)

args = parser.parse_args()

################################################################################
# Argillaクライアントを初期化し、データセットを読み込む
################################################################################

client = rg.Argilla(api_key=args.argilla_api_key, api_url=args.argilla_api_url)
dataset = client.datasets(args.dataset_path)

################################################################################
# Argillaレコードを処理
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
# Hugging Faceデータセットを作成し、Hubにプッシュ
################################################################################

hf_dataset = Dataset.from_list(dataset_rows)
hf_dataset.push_to_hub(repo_id=args.dataset_repo_id)

print(f"データセットが{args.dataset_repo_id}に正常にプッシュされました")
