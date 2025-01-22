# インストラクションチューニング

このモジュールでは、言語モデルのインストラクションチューニングのプロセスをガイドします。インストラクションチューニングとは、特定のタスクに対してモデルを適応させるために、特定のタスクに関連するデータセットで追加のトレーニングを行うことを指します。このプロセスは、特定のタスクにおけるモデルのパフォーマンスを向上させるのに役立ちます。

このモジュールでは、2つのトピックを探ります：1) チャットテンプレートと2) 教師あり微調整

## 1️⃣ チャットテンプレート

チャットテンプレートは、ユーザーとAIモデル間のインタラクションを構造化し、一貫性のある文脈に適した応答を保証します。これらのテンプレートには、システムメッセージや役割に基づくメッセージなどのコンポーネントが含まれます。詳細については、[チャットテンプレート](./chat_templates.md)セクションを参照してください。

## 2️⃣ 教師あり微調整

教師あり微調整（SFT）は、事前トレーニングされた言語モデルを特定のタスクに適応させるための重要なプロセスです。これは、ラベル付きの例を含む特定のタスクのデータセットでモデルをトレーニングすることを含みます。SFTの詳細なガイド、重要なステップ、およびベストプラクティスについては、[教師あり微調整](./supervised_fine_tuning.md)ページを参照してください。

## 演習ノートブック

| タイトル | 説明 | 演習 | リンク | Colab |
|--------|-------------|-----------|--------|-------|
| チャットテンプレート | SmolLM2を使用してチャットテンプレートを使用し、チャットml形式のデータセットを処理する方法を学びます | 🐢 `HuggingFaceTB/smoltalk`データセットをchatml形式に変換 <br> 🐕 `openai/gsm8k`データセットをchatml形式に変換 | [ノートブック](./notebooks/chat_templates_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/1_instruction_tuning/notebooks/chat_templates_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| 教師あり微調整 | SFTTrainerを使用してSmolLM2を微調整する方法を学びます | 🐢 `HuggingFaceTB/smoltalk`データセットを使用 <br> 🐕 `bigcode/the-stack-smol`データセットを試す <br> 🦁 実際の使用ケースに関連するデータセットを選択 | [ノートブック](./notebooks/sft_finetuning_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/1_instruction_tuning/notebooks/sft_finetuning_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

## 参考文献

- [Transformersのチャットテンプレートに関するドキュメント](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [TRLの教師あり微調整スクリプト](https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py)
- [TRLの`SFTTrainer`](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [直接選好最適化に関する論文](https://arxiv.org/abs/2305.18290)
- [TRLを使用した教師あり微調整](https://huggingface.co/docs/trl/main/en/tutorials/supervised_fine_tuning)
- [ChatMLとHugging Face TRLを使用したGoogle Gemmaの微調整方法](https://www.philschmid.de/fine-tune-google-gemma)
- [LLMを微調整してペルシャ語の商品カタログをJSON形式で生成する方法](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_to_generate_persian_product_catalogs_in_json_format)
