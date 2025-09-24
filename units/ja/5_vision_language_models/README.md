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
