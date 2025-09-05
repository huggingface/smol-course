# ビジョン言語モデル

## 1. VLMの使用

ビジョン言語モデル（VLM）は、画像キャプション生成、視覚質問応答、マルチモーダル推論などのタスクを可能にするために、テキストと並行して画像入力を処理します。

典型的なVLMアーキテクチャは、視覚的特徴を抽出する画像エンコーダ、視覚的およびテキスト表現を整列させるプロジェクション層、およびテキストを処理または生成する言語モデルで構成されます。これにより、モデルは視覚要素と言語概念の間の接続を確立できます。

VLMは、使用ケースに応じてさまざまな構成で使用できます。ベースモデルは一般的なビジョン言語タスクを処理し、チャット最適化されたバリアントは会話型インタラクションをサポートします。一部のモデルには、視覚的証拠に基づいて予測を行うための追加コンポーネントや、物体検出などの特定のタスクに特化したコンポーネントが含まれています。

VLMの技術的な詳細と使用方法については、[VLMの使用](./vlm_usage.md)ページを参照してください。

## 2. VLMのファインチューニング

VLMのファインチューニングは、特定のタスクを実行するため、または特定のデータセットで効果的に動作するように、事前トレーニングされたモデルを適応させるプロセスです。このプロセスは、モジュール1および2で紹介されたように、教師ありファインチューニング、好みの最適化、またはその両方を組み合わせたハイブリッドアプローチなどの方法論に従うことができます。

コアツールと技術はLLMで使用されるものと似ていますが、VLMのファインチューニングには、画像のデータ表現と準備に特に注意を払う必要があります。これにより、モデルが視覚データとテキストデータの両方を効果的に統合および処理し、最適なパフォーマンスを発揮できるようになります。デモモデルであるSmolVLMは、前のモジュールで使用された言語モデルよりも大幅に大きいため、効率的なファインチューニング方法を探ることが重要です。量子化やPEFTなどの技術を使用することで、プロセスをよりアクセスしやすく、コスト効果の高いものにし、より多くのユーザーがモデルを試すことができます。

VLMのファインチューニングに関する詳細なガイダンスについては、[VLMのファインチューニング](./vlm_finetuning.md)ページを参照してください。

## 演習ノートブック

| タイトル | 説明 | 演習 | リンク | Colab |
|-------|-------------|----------|------|-------|
| VLMの使用 | 事前トレーニングされたVLMをさまざまなタスクに使用する方法を学ぶ | 🐢 画像を処理する<br>🐕 バッチ処理で複数の画像を処理する<br>🦁 ビデオ全体を処理する | [ノートブック](./notebooks/vlm_usage_sample.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/5_vision_language_models/notebooks/vlm_usage_sample.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| VLMのファインチューニング | タスク固有のデータセットに対して事前トレーニングされたVLMをファインチューニングする方法を学ぶ | 🐢 基本的なデータセットを使用してファインチューニングする<br>🐕 新しいデータセットを試す<br>🦁 代替のファインチューニング方法を試す | [ノートブック](./notebooks/vlm_sft_sample.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/5_vision_language_models/notebooks/vlm_sft_sample.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

## 参考文献
- [Hugging Face Learn: Supervised Fine-Tuning VLMs](https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl)
- [Hugging Face Learn: Supervised Fine-Tuning SmolVLM](https://huggingface.co/learn/cookbook/fine_tuning_smol_vlm_sft_trl)
- [Hugging Face Learn: Preference Optimization Fine-Tuning SmolVLM](https://huggingface.co/learn/cookbook/fine_tuning_vlm_dpo_smolvlm_instruct)
- [Hugging Face Blog: Preference Optimization for VLMs](https://huggingface.co/blog/dpo_vlm)
- [Hugging Face Blog: Vision Language Models](https://huggingface.co/blog/vlms)
- [Hugging Face Blog: SmolVLM](https://huggingface.co/blog/smolvlm)
- [Hugging Face Model: SmolVLM-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)
- [CLIP: Learning Transferable Visual Models from Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [Align Before Fuse: Vision and Language Representation Learning with Momentum Distillation](https://arxiv.org/abs/2107.07651)
