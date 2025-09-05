# パラメータ効率の良い微調整 (PEFT)

言語モデルが大きくなるにつれて、従来の微調整はますます困難になります。1.7Bパラメータのモデルの完全な微調整には、かなりのGPUメモリが必要であり、別々のモデルコピーを保存することが高価であり、モデルの元の能力を破壊的に忘れるリスクがあります。パラメータ効率の良い微調整（PEFT）メソッドは、モデルパラメータの小さなサブセットのみを変更し、ほとんどのモデルを固定したままにすることで、これらの課題に対処します。

従来の微調整は、トレーニング中にすべてのモデルパラメータを更新しますが、これは大規模なモデルには実用的ではありません。PEFTメソッドは、元のモデルサイズの1％未満のパラメータを使用してモデルを適応させるアプローチを導入します。この劇的な学習可能なパラメータの削減により、次のことが可能になります：

- 限られたGPUメモリを持つ消費者向けハードウェアでの微調整
- 複数のタスク固有の適応を効率的に保存
- 低データシナリオでのより良い一般化
- より速いトレーニングと反復サイクル

## 利用可能なメソッド

このモジュールでは、2つの人気のあるPEFTメソッドを紹介します：

### 1️⃣ LoRA (低ランク適応)

LoRAは、効率的なモデル適応のためのエレガントなソリューションを提供する最も広く採用されているPEFTメソッドとして浮上しました。モデル全体を変更する代わりに、**LoRAはモデルの注意層に学習可能な行列を注入します。**このアプローチは、通常、完全な微調整と比較して約90％の学習可能なパラメータを削減しながら、同等の性能を維持します。[LoRA (低ランク適応)](./lora_adapters.md)セクションでLoRAを詳しく見ていきます。
 
### 2️⃣ プロンプトチューニング

プロンプトチューニングは、モデルの重みを変更するのではなく、**入力に学習可能なトークンを追加する**さらに軽量なアプローチを提供します。プロンプトチューニングはLoRAほど人気はありませんが、モデルを新しいタスクやドメインに迅速に適応させるための便利な技術です。[プロンプトチューニング](./prompt_tuning.md)セクションでプロンプトチューニングを詳しく見ていきます。

## 演習ノートブック

| タイトル | 説明 | 演習 | リンク | Colab |
|-------|-------------|----------|------|-------|
| LoRA微調整 | LoRAアダプタを使用してモデルを微調整する方法を学ぶ | 🐢 LoRAを使用してモデルをトレーニング<br>🐕 異なるランク値で実験<br>🦁 完全な微調整と性能を比較 | [ノートブック](./notebooks/finetune_sft_peft.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/3_parameter_efficient_finetuning/notebooks/finetune_sft_peft.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| LoRAアダプタの読み込み | トレーニングされたLoRAアダプタを読み込んで使用する方法を学ぶ | 🐢 事前学習されたアダプタを読み込む<br>🐕 アダプタをベースモデルと統合<br>🦁 複数のアダプタ間を切り替える | [ノートブック](./notebooks/load_lora_adapter.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/3_parameter_efficient_finetuning/notebooks/load_lora_adapter.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
<!-- | プロンプトチューニング | プロンプトチューニングを実装する方法を学ぶ | 🐢 ソフトプロンプトをトレーニング<br>🐕 異なる初期化戦略を比較<br>🦁 複数のタスクで評価 | [ノートブック](./notebooks/prompt_tuning_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/3_parameter_efficient_finetuning/notebooks/prompt_tuning_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | -->

## リソース
- [PEFTドキュメント](https://huggingface.co/docs/peft)
- [LoRA論文](https://arxiv.org/abs/2106.09685)
- [QLoRA論文](https://arxiv.org/abs/2305.14314)
- [プロンプトチューニング論文](https://arxiv.org/abs/2104.08691)
- [Hugging Face PEFTガイド](https://huggingface.co/blog/peft)
- [2024年にHugging FaceでLLMを微調整する方法](https://www.philschmid.de/fine-tune-llms-in-2024-with-trl) 
- [TRL](https://huggingface.co/docs/trl/index)
