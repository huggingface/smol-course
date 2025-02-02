# 推論

推論は、訓練された言語モデルを使用して予測や応答を生成するプロセスです。推論は一見簡単に思えるかもしれませんが、スケールでモデルを効率的にデプロイするには、パフォーマンス、コスト、信頼性などのさまざまな要素を慎重に考慮する必要があります。大規模言語モデル（LLM）は、そのサイズと計算要件のために独自の課題を提示します。

ここでは、[`transformers`](https://huggingface.co/docs/transformers/index)ライブラリと[`text-generation-inference`](https://github.com/huggingface/text-generation-inference)という2つの人気のあるLLM推論フレームワークを使用して、シンプルなアプローチとプロダクション対応のアプローチの両方を探ります。プロダクションデプロイメントについては、最適化されたサービング機能を提供するText Generation Inference（TGI）に焦点を当てます。

## モジュール概要

LLM推論は、開発とテストのためのシンプルなパイプラインベースの推論と、プロダクションデプロイメントのための最適化されたサービングソリューションの2つの主要なアプローチに分類できます。まず、シンプルなパイプラインアプローチから始め、プロダクション対応のソリューションに進みます。

## コンテンツ

### 1. [基本的なパイプライン推論](./pipeline_inference.md)

Hugging Face Transformersパイプラインを使用した基本的な推論方法を学びます。パイプラインの設定、生成パラメータの構成、およびローカル開発のベストプラクティスについて説明します。パイプラインアプローチは、プロトタイピングや小規模なアプリケーションに最適です。[学習を開始](./pipeline_inference.md)。

### 2. [TGIを使用したプロダクション推論](./tgi_inference.md)

Text Generation Inferenceを使用してモデルをプロダクションにデプロイする方法を学びます。最適化されたサービング技術、バッチ処理戦略、およびモニタリングソリューションについて探ります。TGIは、ヘルスチェック、メトリクス、Dockerデプロイメントオプションなどのプロダクション対応機能を提供します。[学習を開始](./text_generation_inference.md)。

### 演習ノートブック

| タイトル | 説明 | 演習 | リンク | Colab |
|-------|-------------|----------|------|-------|
| パイプライン推論 | transformersパイプラインを使用した基本的な推論 | 🐢 基本的なパイプラインの設定 <br> 🐕 生成パラメータの構成 <br> 🦁 シンプルなWebサーバーの作成 | [リンク](./notebooks/basic_pipeline_inference.ipynb) | [Colab](https://githubtocolab.com/huggingface/smol-course/tree/main/7_inference/notebooks/basic_pipeline_inference.ipynb) |
| TGIデプロイメント | TGIを使用したプロダクションデプロイメント | 🐢 TGIでモデルをデプロイ <br> 🐕 パフォーマンス最適化の構成 <br> 🦁 モニタリングとスケーリングの設定 | [リンク](./notebooks/tgi_deployment.ipynb) | [Colab](https://githubtocolab.com/huggingface/smol-course/tree/main/7_inference/notebooks/tgi_deployment.ipynb) |

## リソース

- [Hugging Face Pipeline Tutorial](https://huggingface.co/docs/transformers/en/pipeline_tutorial)
- [Text Generation Inference Documentation](https://huggingface.co/docs/text-generation-inference/en/index)
- [Pipeline WebServer Guide](https://huggingface.co/docs/transformers/en/pipeline_tutorial#using-pipelines-for-a-webserver)
- [TGI GitHub Repository](https://github.com/huggingface/text-generation-inference)
- [Hugging Face Model Deployment Documentation](https://huggingface.co/docs/inference-endpoints/index)
- [vLLM: High-throughput LLM Serving](https://github.com/vllm-project/vllm)
- [Optimizing Transformer Inference](https://huggingface.co/blog/optimize-transformer-inference)
