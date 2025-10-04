# Argilla、Distilabel、LightEvalを使用したドメイン固有の評価

ほとんどの人気のあるベンチマークは、一般的な能力（推論、数学、コード）を評価しますが、より具体的な能力を評価する必要がある場合はどうすればよいでしょうか？

カスタムドメインに関連するモ���ルを評価する必要がある場合はどうすればよいでしょうか？（例えば、金融、法務、医療のユースケース）

このチュートリアルでは、[Argilla](https://github.com/argilla-io/argilla)、[distilabel](https://github.com/argilla-io/distilabel)、および[LightEval](https://github.com/huggingface/lighteval)を使用して、関連するデータを作成し、サンプルに注釈を付け、モデルを評価するための完全なパイプラインを示します。例として、複数のドキュメントから試験問題を生成することに焦点を当てます。

## プロジェクト構造

このプロセスでは、データセットの生成、注釈の追加、評価のためのサンプルの抽出、実際のモデル評価の4つのステップを実行します。各ステップにはスクリプトがあります。

| スクリプト名 | 説明 |
|-------------|-------------|
| generate_dataset.py | 指定された言語モデルを使用して複数のテキストドキュメントから試験問題を生成します。 |
| annotate_dataset.py | 生成された試験問題の手動注釈のためのArgillaデータセットを作成します。 |
| create_dataset.py | Argillaから注釈付きデータを処理し、Hugging Faceデータセットを作成します。 |
| evaluation_task.py | 試験問題データセットの評価のためのカスタムLightEvalタスクを定義します。 |

## ステップ

### 1. データセットの生成

`generate_dataset.py`スクリプトは、distilabelライブラリを使用して複数のテキストドキュメントに基づいて試験問題を生成します。指定されたモデル（デフォルト：Meta-Llama-3.1-8B-Instruct）を使用して質問、正しい回答、および誤った回答（ディストラクター）を作成します。独自のデータサンプルを追加し、異なるモデルを使用することもできます。

生成を実行するには：

```sh
python generate_dataset.py --input_dir path/to/your/documents --model_id your_model_id --output_path output_directory
```

これにより、入力ディレクトリ内のすべてのドキュメントに対して生成された試験問題を含む[Distiset](https://distilabel.argilla.io/dev/sections/how_to_guides/advanced/distiset/)が作成されます。

### 2. データセットの注釈

`annotate_dataset.py`スクリプトは、生成された質問を取得し、注釈のためのArgillaデータセットを作成します。データセットの構造を設定し、生成された質問と回答をランダムな順序で入力します。これにより、バイアスを避けることができます。Argillaでは、正しい回答が提案として表示されます。

LLMからの提案された正しい回答がランダムな順序で表示され、正しい回答を承認するか、別の回答を選択できます。このプロセスの所要時間は、評価データセットの規模、ドメインデータの複雑さ、およびLLMの品質によって異なります。例えば、Llama-3.1-70B-Instructを使用して、転移学習のドメインで150サンプルを1時間以内に作成できました。

注釈プロセスを実行するには：

```sh
python annotate_dataset.py --dataset_path path/to/distiset --output_dataset_name argilla_dataset_name
```

これにより、手動レビューと注釈のためのArgillaデータセットが作成されます。

![argilla_dataset](./images/domain_eval_argilla_view.png)

Argillaを使用していない場合は、この[クイックスタートガイド](https://docs.argilla.io/latest/getting_started/quickstart/)に従ってローカルまたはスペースにデプロイしてください。

### 3. データセットの作成

`create_dataset.py`スクリプトは、Argillaから注釈付きデータを処理し、Hugging Faceデータセットを作成します。提案された回答と手動で注釈された回答の両方を処理します。このスクリプトは、質問、可能な回答、および正しい回答の列名を含むデータセットを作成します。最終データセットを作成するには：

```sh
huggingface_hub login
python create_dataset.py --dataset_path argilla_dataset_name --dataset_repo_id your_hf_repo_id
```

これにより、指定されたリポジトリにデータセットがHugging Face Hubにプッシュされます。サンプルデータセットは[こちら](https://huggingface.co/datasets/burtenshaw/exam_questions/viewer/default/train)で確認できます。データセットのプレビューは次のようになります：

![hf_dataset](./images/domain_eval_dataset_viewer.png)

### 4. 評価タスク

`evaluation_task.py`スクリプトは、試験問題データセットの評価のためのカスタムLightEvalタスクを定義します。プロンプト関数、カスタム精度メトリック、およびタスク構成が含まれます。

カスタム試験問題タスクを使用してlightevalでモデルを評価するには：

```sh
lighteval accelerate \
    --model_args "pretrained=HuggingFaceH4/zephyr-7b-beta" \
    --tasks "community|exam_questions|0|0" \
    --custom_tasks domain-eval/evaluation_task.py \
    --output_dir "./evals"
```

詳細なガイドはlighteval wikiで確認できます：

- [カスタムタスクの作成](https://github.com/huggingface/lighteval/wiki/Adding-a-Custom-Task)
- [カスタムメトリックの作成](https://github.com/huggingface/lighteval/wiki/Adding-a-New-Metric)
- [既存のメトリックの使用](https://github.com/huggingface/lighteval/wiki/Metric-List)
