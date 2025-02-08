import argparse
import json
from random import choices, sample

import argilla as rg
from distilabel.distiset import Distiset

################################################################################
# スクリプトのパラメータ
################################################################################

parser = argparse.ArgumentParser(
    description="Argillaを使用して試験問題データセットに注釈を付けます。"
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
    help="試験問題データセットのパス",
)
parser.add_argument(
    "--dataset_config",
    type=str,
    default="default",
    help="データセットの構成",
)
parser.add_argument(
    "--dataset_split",
    type=str,
    default="train",
    help="使用するデータセットの分割",
)
parser.add_argument(
    "--output_dataset_name",
    type=str,
    default="exam_questions",
    help="出力Argillaデータセットの名前",
)

args = parser.parse_args()

################################################################################
# 検証のためのフィードバックタスクを使用してArgillaデータセットを作成
################################################################################

client = rg.Argilla(api_key=args.argilla_api_key, api_url=args.argilla_api_url)

if client.datasets(args.output_dataset_name):
    print(f"既存のデータセット '{args.output_dataset_name}' を削除します")
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
            description="質問を改善できますか？",
        ),
        rg.TextQuestion(
            name="improved_answer",
            description="最良の回答を改善できますか？",
        ),
    ],
)

dataset = rg.Dataset(settings=settings, name=args.output_dataset_name)
dataset.create()

################################################################################
# Distisetを読み込み、Argillaデータセットにレコードを処理して追加
# バイアスを避けるために質問がランダムな順序で表示されることを確認します
# ただし、Argilla UIでは正しい回答を提案として表示します。
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
    f"データセット '{args.output_dataset_name}' がArgillaに作成され、入力されました。"
)
