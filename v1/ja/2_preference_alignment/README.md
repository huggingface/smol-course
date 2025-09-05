# 選好の整合

このモジュールでは、言語モデルを人間の選好に合わせるための技術について説明します。教師あり微調整（SFT）がモデルにタスクを学習させるのに役立つ一方で、選好の整合は出力が人間の期待や価値観に一致するようにします。

## 概要

選好の整合の典型的な方法には、複数のステージが含まれます：
1. 教師あり微調整（SFT）でモデルを特定のドメインに適応させる。
2. 選好の整合（RLHFやDPOなど）で応答の質を向上させる。

ORPOのような代替アプローチは、指示調整と選好の整合を単一のプロセスに統合します。ここでは、DPOとORPOのアルゴリズムに焦点を当てます。

さまざまな整合技術について詳しく知りたい場合は、[Argillaのブログ](https://argilla.io/blog/mantisnlp-rlhf-part-8)を参照してください。

### 1️⃣ 直接選好最適化（DPO）

直接選好最適化（DPO）は、選好データを使用してモデルを直接最適化することで、選好の整合を簡素化します。このアプローチは、別個の報酬モデルや複雑な強化学習を必要とせず、従来のRLHFよりも安定して効率的です。詳細については、[直接選好最適化（DPO）のドキュメント](./dpo.md)を参照してください。

### 2️⃣ 選好確率比最適化（ORPO）

ORPOは、指示調整と選好の整合を単一のプロセスに統合する新しいアプローチを導入します。これは、負の対数尤度損失とトークンレベルのオッズ比項を組み合わせて標準的な言語モデリングの目的を修正します。このアプローチは、単一のトレーニングステージ、参照モデル不要のアーキテクチャ、および計算効率の向上を提供します。ORPOは、さまざまなベンチマークで印象的な結果を示しており、従来の方法と比較してAlpacaEvalで優れたパフォーマンスを示しています。詳細については、[選好確率比最適化（ORPO）のドキュメント](./orpo.md)を参照してください。

## 実習ノートブック

| タイトル | 説明 | 実習内容 | リンク | Colab |
|-------|-------------|----------|------|-------|
| DPOトレーニング | 直接選好最適化を使用してモデルを��レーニングする方法を学ぶ | 🐢 AnthropicのHH-RLHFデータセットを使用してモデルをトレーニングする<br>🐕 独自の選好データセットを使用する<br>🦁 さまざまな選好データセットとモデルサイズで実験する | [ノートブック](./notebooks/dpo_finetuning_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/2_preference_alignment/notebooks/dpo_finetuning_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| ORPOトレーニング | 選好確率比最適化を使用してモデルをトレーニングする方法を学ぶ | 🐢 指示と選好データを使用してモデルをトレーニングする<br>🐕 損失の重みを変えて実験する<br>🦁 ORPOとDPOの結果を比較する | [ノートブック](./notebooks/orpo_finetuning_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/2_preference_alignment/notebooks/orpo_finetuning_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

## リソース

- [TRLのドキュメント](https://huggingface.co/docs/trl/index) - DPOを含むさまざまな整合技術を実装するためのTransformers Reinforcement Learning（TRL）ライブラリのドキュメント。
- [DPO論文](https://arxiv.org/abs/2305.18290) - 人間のフィードバックを用いた強化学習の代替として、選好データを使用して言語モデルを直接最適化するシンプルなアプローチを紹介する論文。
- [ORPO論文](https://arxiv.org/abs/2403.07691) - 指示調整と選好の整合を単一のトレーニングステージに統合する新しいアプローチを紹介する論文。
- [ArgillaのRLHFガイド](https://argilla.io/blog/mantisnlp-rlhf-part-8/) - RLHF、DPOなどのさまざまな整合技術とその実践的な実装について説明するガイド。
- [DPOに関するブログ記事](https://huggingface.co/blog/dpo-trl) - TRLライブラリを使用してDPOを実装する方法についての実践ガイド。コード例とベストプラクティスが含まれています。
- [TRLのDPOスクリプト例](https://github.com/huggingface/trl/blob/main/examples/scripts/dpo.py) - TRLライブラリを使用してDPOトレーニングを実装する方法を示す完全なスクリプト例。
- [TRLのORPOスクリプト例](https://github.com/huggingface/trl/blob/main/examples/scripts/orpo.py) - TRLライブラリを使用してORPOトレーニングを実装するためのリファレンス実装。詳細な設定オプションが含まれています。
- [Hugging Faceの整合ハンドブック](https://github.com/huggingface/alignment-handbook) - SFT、DPO、RLHFなどのさまざまな技術を使用して言語モデルを整合させるためのガイドとコード。
