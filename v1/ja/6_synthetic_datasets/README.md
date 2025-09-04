# 合成データセット

合成データは、実際の使用状況を模倣する人工的に生成されたデータです。データセットを拡張または強化することで、データの制限を克服することができます。合成データはすでにいくつかのユースケースで使用されていましたが、大規模な言語モデルは、言語モデルの事前および事後トレーニング、および評価のための合成データセットをより一般的にしました。

私たちは、検証済みの研究論文に基づいた迅速で信頼性が高くスケーラブルなパイプラインを必要とするエンジニアのための合成データとAIフィードバックのフレームワークである[`distilabel`](https://distilabel.argilla.io/latest/)を使用します。パッケージとベストプラクティスの詳細については、[ドキュメント](https://distilabel.argilla.io/latest/)を参照してください。

## モジュール概要

言語モデルの合成データは、インストラクション、嗜好、批評の3つの分類に分類できます。私たちは、インストラクションチューニングと嗜好調整のためのデータセットの生成に焦点を当てます。両方のカテゴリでは、既存のデータをモデルの批評とリライトで改善するための第3のカテゴリの側面もカバーします。

![合成データの分類](./images/taxonomy-synthetic-data.png)

## コンテンツ

### 1. [インストラクションデータセット](./instruction_datasets.md)

インストラクションチューニングのためのインストラクションデータセットの生成方法を学びます。基本的なプロンプトを使用してインストラクションチューニングデータセットを作成する方法や、論文から得られたより洗練された技術を使用する方法を探ります。SelfInstructやMagpieのような方法を使用して、インコンテキスト学習のためのシードデータを使用してインストラクションチューニングデータセットを作成できます。さらに、EvolInstructを通じたインストラクションの進化についても探ります。[学び始める](./instruction_datasets.md)。

### 2. [嗜好データセット](./preference_datasets.md)

嗜好調整のための嗜好データセットの生成方法を学びます。セクション1で紹介した方法と技術を基に構築し、追加の応答を生成します。次に、EvolQualityプロンプトを使用して応答を改善する方法を学びます。最後に、スコアと批評を生成するUltraFeedbackプロンプトを探り、嗜好ペアを作成します。[学び始める](./preference_datasets.md)。

### 演習ノートブック

| タイトル | 説明 | 演習 | リンク | Colab |
|-------|-------------|----------|------|-------|
| インストラクションデータセット | インストラクションチューニングのためのデータセットを生成する | 🐢 インストラクションチューニングデータセットを生成する <br> 🐕 シードデータを使用してインストラクションチューニングのためのデータセットを生成する <br> 🦁 シードデータとインストラクションの進化を使用してインストラクションチューニングのためのデータセットを生成する | [リンク](./notebooks/instruction_sft_dataset.ipynb) | [Colab](https://githubtocolab.com/huggingface/smol-course/tree/main/6_synthetic_datasets/notebooks/instruction_sft_dataset.ipynb) |
| 嗜好データセット | 嗜好調整のためのデータセットを生成する | 🐢 嗜好調整データセットを生成する <br> 🐕 応答の進化を使用して嗜好調整のためのデータセットを生成する <br> 🦁 応答の進化と批評を使用して嗜好調整のためのデータセットを生成する | [リンク](./notebooks/preference_alignment_dataset.ipynb) | [Colab](https://githubtocolab.com/huggingface/smol-course/tree/main/6_synthetic_datasets/notebooks/preference_alignment_dataset.ipynb) |

## リソース

- [Distilabel Documentation](https://distilabel.argilla.io/latest/)
- [Synthetic Data Generator is UI app](https://huggingface.co/blog/synthetic-data-generator)
- [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)
- [Self-instruct](https://arxiv.org/abs/2212.10560)
- [Evol-Instruct](https://arxiv.org/abs/2304.12244)
- [Magpie](https://arxiv.org/abs/2406.08464)
- [UltraFeedback](https://arxiv.org/abs/2310.01377)
- [Deita](https://arxiv.org/abs/2312.15685)
