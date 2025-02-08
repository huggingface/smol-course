import argparse
import os
from pydantic import BaseModel, Field
from datasets import Dataset
from typing import List

from distilabel.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration


################################################################################
# スクリプトのパラメータ
################################################################################

parser = argparse.ArgumentParser(
    description="ディレクトリ内のテキストファイルから試験問題を生成します。"
)
parser.add_argument(
    "--model_id",
    type=str,
    default="Qwen/Qwen2.5-7B-Instruct",
    help="テキスト生成のためのモデルID",
)
parser.add_argument(
    "--tokenizer_id",
    type=str,
    default="Qwen/Qwen2.5-7B-Instruct",
    help="テキスト生成のためのトークナイザーID",
)
parser.add_argument(
    "--input_dir",
    type=str,
    help="入力テキストファイルを含むディレクトリ",
    default="data",
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=2048,
    help="生成する新しいトークンの最大数",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="exam_questions_output",
    help="生成されたデータセットを保存するディレクトリ",
)

args = parser.parse_args()

################################################################################
# ドキュメントの読み込み
# ドキュメントは入力ディレクトリにあり、各ファイルが同じトピックに関する
# 個別のドキュメントであると仮定します。
################################################################################

# 入力ディレクトリ内のすべてのテキストファイルを処理
documents = []
for filename in os.listdir(args.input_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(args.input_dir, filename)
        with open(file=file_path, mode="r", encoding="utf-8") as file:
            document_content = file.read()
            documents.append(document_content)

# すべてのドキュメント内容から単一のデータセットを作成
dataset = Dataset.from_dict({"document": documents})

################################################################################
# プロンプトの定義
# モデルが正しい出力形式を生成するようにシステムプロンプトを使用します。
# テンプレートを使用してドキュメントをプロンプトに挿入します。
################################################################################

SYSTEM_PROMPT = """\
あなたは学生のための試験問題を作成する専門家です。
提供されたドキュメントに基づいて質問と回答を作成し、
質問に対する正しい回答と、誤っているが妥当な回答のリストを作成してください。
回答は次の形式に従う必要があります：
```
[
    {
        "question": "質問内容",
        "answer": "質問に対する正しい回答",
        "distractors": ["誤った回答1", "誤った回答2", "誤った回答3"]
    },
    ... (必要に応じてさらに質問と回答を追加)
]
```
""".strip()

INSTRUCTION_TEMPLATE = """\
    ドキュメントに関する質問と回答のリストを生成してください。 
    ドキュメント:\n\n{{ instruction }}"""

################################################################################
# 出力構造の定義
# パイプラインの出力のデータモデルを定義し、評価タスクの正しい形式で
# 出力があることを確認します。
################################################################################


class ExamQuestion(BaseModel):
    question: str = Field(..., description="回答すべき質問")
    answer: str = Field(..., description="質問に対する正しい回答")
    distractors: List[str] = Field(
        ..., description="質問に対する誤っているが妥当な回答のリスト"
    )


class ExamQuestions(BaseModel):
    exam: List[ExamQuestion]


################################################################################
# パイプラインの作成
# ドキュメントに基づいて試験問題を生成し、正しい形式で出力する単一のタスクを
# 持つパイプラインを作成します。Hugging Face InferenceEndpointsと
# 引数で指定されたモデルを使用します。
################################################################################

with Pipeline(
    name="Domain-Eval-Questions",
    description="提供されたドキュメントに基づいて試験問題を生成します。",
) as pipeline:
    # テキスト生成タスクの設定
    text_generation = TextGeneration(
        name="exam_generation",
        llm=InferenceEndpointsLLM(
            model_id=args.model_id,
            tokenizer_id=args.model_id,
            api_key=os.environ["HF_TOKEN"],
            structured_output={
                "schema": ExamQuestions.model_json_schema(),
                "format": "json",
            },
        ),
        input_batch_size=8,
        output_mappings={"model_name": "generation_model"},
        input_mappings={"instruction": "document"},
        system_prompt=SYSTEM_PROMPT,
        template=INSTRUCTION_TEMPLATE,
    )


################################################################################
# パイプラインの実行
# すべてのドキュメントに対してパイプラインを実行し、結果を出力パスに保存します。
################################################################################

if __name__ == "__main__":
    # すべてのドキュメントに対してパイプラインを実行
    distiset = pipeline.run(
        parameters={
            "exam_generation": {
                "llm": {
                    "generation_kwargs": {
                        "max_new_tokens": args.max_new_tokens,
                    }
                }
            }
        },
        use_cache=False,
        dataset=dataset,
    )

    distiset.save_to_disk(args.output_path)
