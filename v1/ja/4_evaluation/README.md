# 評価

評価は、言語モデルの開発と展開において重要なステップです。評価は、モデルがさまざまな能力にわたってどれだけうまく機能するかを理解し、改善の余地を特定するのに役立ちます。このモジュールでは、標準ベンチマークとドメイン固有の評価アプローチの両方をカバーし、smolモデルを包括的に評価します。

Hugging Faceが開発した強力な評価ライブラリである[`lighteval`](https://github.com/huggingface/lighteval)を使用します。評価の概念とベストプラクティスについて詳しく知りたい場合は、評価[ガイドブック](https://github.com/huggingface/evaluation-guidebook)を参照してください。

## モジュール概要

徹底した評価戦略は、モデル性能の複数の側面を検討します。質問応答や要約などのタスク固有の能力を評価し、モデルがさまざまなタイプの問題にどれだけうまく対処できるかを理解します。出力の品質を一貫性や事実の正確性などの要素で測定します。安全性評価は、潜在的な有害な出力やバイアスを特定するのに役立ちます。最後に、ドメインの専門知識テストは、ターゲット分野でのモデルの専門知識を検証します。

## コンテンツ

### 1️⃣ [自動ベンチマーク](./automatic_benchmarks.md)

標準化されたベンチマークとメトリクスを使用してモデルを評価する方法を学びます。MMLUやTruthfulQAなどの一般的なベンチマークを探求し、主要な評価メトリクスと設定を理解し、再現可能な評価のベストプラクティスをカバーします。

### 2️⃣ [カスタムドメイン評価](./custom_evaluation.md)

特定のユースケースに合わせた評価パイプラインを作成する方法を学びます。カスタム評価タスクの設計、専門的なメトリクスの実装、要件に合った評価データセットの構築について説明します。

### 3️⃣ [ドメイン評価プロジェクト](./project/README.md)

ドメイン固有の評価パイプラインを構築する完全な例を紹介します。評価データセットの生成、データ注釈のためのArgillaの使用、標準化されたデータセットの作成、LightEvalを使用したモデルの評価方法を学びます。

### 演習ノートブック

| タイトル | 説明 | 演習 | リンク | Colab |
|-------|-------------|----------|------|-------|
| LLMの評価と分析 | LightEvalを使用して特定のドメインでモデルを評価および比較する方法を学びます | 🐢 医療ドメインタスクを使用してモデルを評価する <br> 🐕 異なるMMLUタスクで新しいドメイン評価を作成する <br> 🦁 ドメイン固有のカスタム評価タスクを作成する | [ノートブック](./notebooks/lighteval_evaluate_and_analyse_your_LLM.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/4_evaluation/notebooks/lighteval_evaluate_and_analyse_your_LLM.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

## リソース

- [評価ガイドブック](https://github.com/huggingface/evaluation-guidebook) - LLM評価の包括的なガイド
- [LightEvalドキュメント](https://github.com/huggingface/lighteval) - LightEvalライブラリの公式ドキュメント
- [Argillaドキュメント](https://docs.argilla.io) - Argillaアノテーションプラットフォームについて学ぶ
- [MMLU論文](https://arxiv.org/abs/2009.03300) - MMLUベンチマークを説明する論文
- [カスタムタスクの作成](https://github.com/huggingface/lighteval/wiki/Adding-a-Custom-Task)
- [カスタムメトリクスの作成](https://github.com/huggingface/lighteval/wiki/Adding-a-New-Metric)
- [既存のメトリクスの使用](https://github.com/huggingface/lighteval/wiki/Metric-List)
