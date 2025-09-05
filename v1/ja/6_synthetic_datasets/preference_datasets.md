# 嗜好データセットの生成

[嗜好調整の章](../2_preference_alignment/README.md)では、直接嗜好最適化について学びました。このセクションでは、DPOのような方法のための嗜好データセットを生成する方法を探ります。[インストラクションデータセットの生成](./instruction_datasets.md)で紹介した方法を基に構築します。さらに、基本的なプロンプトを使用してデータセットに追加の完了を追加する方法や、EvolQualityを使用して応答の品質を向上させる方法を示します。最後に、UltraFeedbackを使用してスコアと批評を生成する方法を示します。

## 複数の完了を作成する

嗜好データは、同じ`インストラクション`に対して複数の`完了`を持つデータセットです。モデルにプロンプトを与えて追加の`完了`を生成することで、データセットにより多くの`完了`を追加できます。この際、2つ目の完了が全体的な品質や表現において最初の完了とあまり似ていないことを確認する必要があります。これは、モデルが明確な嗜好に最適化される必要があるためです。通常、`選ばれた`と`拒否された`と呼ばれる完了のどちらが好まれるかを知りたいのです。[スコアの作成セクション](#creating-scores)で、選ばれた完了と拒否された完了を決定する方法について詳しく説明します。

### モデルプーリング

異なるモデルファミリーからモデルを使用して2つ目の完了を生成することができます。これをモデルプーリングと呼びます。2つ目の完了の品質をさらに向上させるために、`温度`を調整するなど、異なる生成引数を使用することができます。最後に、異なるプロンプトテンプレートやシステムプロンプトを使用して2つ目の完了を生成し、特定の特性に基づいて多様性を確保することができます。理論的には、異なる品質の2つのモデルを使用し、より良いものを`選ばれた`完了として使用することができます。

まず、[Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)と[HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)モデルを`transformers`統合の`distilabel`ライブラリを使用してロードします。これらのモデルを使用して、与えられた`プロンプト`に対して2つの合成`応答`を作成します。`LoadDataFromDicts`、`TextGeneration`、および`GroupColumns`を使用して別のパイプラインを作成します。最初にデータをロードし、次に2つの生成ステップを使用し、最後に結果をグループ化します。ステップを接続し、`>>`演算子と`[]`を使用してデータをパイプラインに流します。これは、前のステップの出力を次のステップの入力として使用することを意味します。

```python
from distilabel.llms import TransformersLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import GroupColumns, LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration

with Pipeline() as pipeline:
    data = LoadDataFromDicts(data=[{"instruction": "合成データとは何ですか？"}])
    llm_a = TransformersLLM(model="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    gen_a = TextGeneration(llm=llm_a)
    llm_b = TransformersLLM(model="Qwen/Qwen2.5-1.5B-Instruct")
    gen_b = TextGeneration(llm=llm_b)
    group = GroupColumns(columns=["generation"])
    data >> [gen_a, gen_b] >> group

if __name__ == "__main__":
    distiset = pipeline.run()
    print(distiset["default"]["train"]["grouped_generation"][0])
# {[
#   '合成データは、実際の使用状況を模倣する人工的に生成されたデータです。',
#   '合成データとは、人工的に生成されたデータを指します。'
# ]}
```

ご覧のとおり、与えられた`プロンプト`に対して2つの合成`完了`があります。生成ステップを特定の`システムプロンプト`で初期化するか、`TransformersLLM`に生成引数を渡すことで、多様性をさらに向上させることができます。次に、EvolQualityを使用して`完了`の品質を向上させる方法を見てみましょう。

### EvolQuality

EvolQualityは、[EvolInstruct](./instruction_datasets.md#evolinstruct)に似ていますが、入力`プロンプト`の代わりに`完了`を進化させます。このタスクは、`プロンプト`と`完了`の両方を受け取り、一連の基準に基づいて`完了`をより良いバージョンに進化させます。このより良いバージョンは、役立ち度、関連性、深化、創造性、または詳細の基準に基づいて定義されます。これにより、データセットに追加の`完了`を自動的に生成できます。理論的には、進化したバージョンが元の完了よりも優れていると仮定し、それを`選ばれた`完了として使用することができます。

プロンプトは[distilabelで実装されています](https://github.com/argilla-io/distilabel/tree/main/src/distilabel/steps/tasks/evol_quality)が、簡略化されたバージョンは以下の通りです：

```bash
応答リライターとして行動してください。
与えられたプロンプトと応答を、より良いバージョンに書き換えてください。
以下の基準に基づいてプロンプトを複雑化してください：
{{ criteria }}

# プロンプト
{{ input }}

# 応答
{{ output }}

# 改善された応答
```

これを使用するには、[EvolQualityクラス](https://distilabel.argilla.io/dev/components-gallery/tasks/evolquality/)に`llm`を渡す必要があります。[モデルプーリングセクション](#model-pooling)の合成`プロンプト`と`完了`を使用して、より良いバージョンに進化させてみましょう。この例では、1世代だけ進化させます。

```python
from distilabel.llms import TransformersLLM
from distilabel.steps.tasks import EvolQuality

llm = TransformersLLM(model="HuggingFaceTB/SmolLM2-1.7B-Instruct")
evol_quality = EvolQuality(llm=llm, num_evolutions=1)
evol_quality.load()

instruction = "合成データとは何ですか？"
completion = "合成データは、実際の使用状況を模倣する人工的に生成されたデータです。"

next(evol_quality.process([{
    "instruction": instruction,
    "response": completion
}]))
# 手動プロンプトを使用して合成データを生成するプロセスは何ですか？
```

`応答`はより複雑で、`インストラクション`に特化しています。これは良いスタートですが、EvolInstructで見たように、進化した生成物が常に優れているわけではありません。したがって、データセットの品質を確保するために追加の評価技術を使用することが重要です。次のセクションでこれを探ります。

## スコアの作成

スコアは、ある応答が他の応答よりもどれだけ好まれるかを測定するものです。一般的に、これらのスコアは絶対的、主観的、または相対的です。このコースでは、最初の2つに焦点を当てます。これらは嗜好データセットの作成に最も価値があるためです。このスコアリングは、評価技術と重なる部分があり、[評価の章](../3_evaluation/README.md)で見たように、通常はより大きなモデルが必要です。

### UltraFeedback

UltraFeedbackは、与えられた`プロンプト`とその`完了`に対してスコアと批評を生成する技術です。

スコアは、一連の基準に基づいて`完了`の品質を評価します。基準には、`役立ち度`、`関連性`、`深化`、`創造性`の4つの細かい基準があります。これらは便利ですが、一般的には全体的な基準を使用するのが良いスタートです。これにより、スコアの生成プロセスが簡略化されます。スコアは、どの`完了`が`選ばれた`もので、どれが`拒否された`ものであるかを決定するために使用できます。絶対的なスコアであるため、データセットの外れ値を見つけるための興味深いフィルターとしても使用できます。たとえば、最悪の完了や差が大きいペアを見つけることができます。

批評はスコアの理由を提供するために追加されます。これにより、スコアの違いを理解するための追加のコンテキストが提供されます。言語モデルは広範な批評を生成しますが、これは非常に便利ですが、スコアを表す単一のトークンを生成するよりもコストと複雑さが増します。

プロンプトは[distilabelで実装されています](https://github.com/argilla-io/distilabel/tree/main/src/distilabel/steps/tasks/templates/ultrafeedback)が、簡略化されたバージョンは以下の通りです：

```bash
モデルの出力をさまざまな基準に基づいて評価してください：役立ち度、関連性、深化、創造性
上記の要因に基づいて全体的な評価を提供する役割を果たします。
出力の全体的な品質を1から5のスコアで評価してください。

以下の形式で回答してください：スコア - 理由

# 入力
{{ input }}

# 応答
{{ output }}

# スコア - 理由
```

これを使用するには、[UltraFeedbackクラス](https://distilabel.argilla.io/dev/components-gallery/tasks/ultrafeedback/)に`llm`を渡す必要があります。[モデルプーリングセクション](#model-pooling)の合成`プロンプト`と`完了`を評価してみましょう。

```python
from distilabel.llms import TransformersLLM
from distilabel.steps.tasks import UltraFeedback

llm = TransformersLLM(model="HuggingFaceTB/SmolLM2-1.7B-Instruct")
ultrafeedback = UltraFeedback(llm=llm)
ultrafeedback.load()

instruction = "合成データとは何ですか？"
completion_a = "合成データは、実際の使用状況を模倣する人工的に生成されたデータです。"
completion_b = "合成データとは、人工的に生成されたデータを指します。"

next(ultrafeedback.process([{
    "instruction": instruction,
    "generations": [completion_a, completion_b]
}]))
# [
#     {
#         'ratings': [4, 5],
#         'rationales': ['could have been more specific', 'good definition'],
#     }
# ]
```

## ベストプラクティス

- 全体的なスコアは、批評や特定のスコアよりも安価で生成が容易です
- スコアや批評を生成するために大きなモデルを使用する
- スコアや批評を生成するために多様なモデルセットを使用する
- `system_prompt`やモデルの構成を繰り返し改善する

## 次のステップ

👨🏽‍💻 コード -[演習ノートブック](./notebooks/preference_dpo_dataset.ipynb)でインストラクションチューニングのためのデータセットを生成する

## 参考文献

- [Distilabel Documentation](https://distilabel.argilla.io/latest/)
- [Deita](https://arxiv.org/abs/2312.15685)
- [UltraFeedback](https://arxiv.org/abs/2310.01377)
