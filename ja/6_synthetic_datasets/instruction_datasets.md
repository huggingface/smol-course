# インストラクションデータセットの生成

[インストラクションチューニングの章](../1_instruction_tuning/README.md)では、教師あり微調整によるモデルの微調整について学びました。このセクションでは、SFTのためのインストラクションデータセットの生成方法を探ります。基本的なプロンプトを使用してインストラクションチューニングデータセットを作成する方法や、論文から得られたより洗練された技術を使用する方法を探ります。インストラクションチューニングデータセットは、SelfInstructやMagpieのような方法を使用して、インコンテキスト学習のためのシードデータを使用して作成できます。さらに、EvolInstructを通じたインストラクションの進化についても探ります。最後に、distilabelパイプラインを使用してインストラクションチューニングのためのデータセットを生成する方法を探ります。

## プロンプトからデータへ

合成データは一見複雑に見えますが、モデルから知識を抽出するための効果的なプロンプトを作成することに簡略化できます。つまり、特定のタスクのためのデータを生成する方法と考えることができます。課題は、データが多様で代表的であることを保証しながら、効果的にプロンプトを作成することです。幸いなことに、多くの論文がこの問題を探求しており、このコースではいくつかの有用なものを探ります。まずは、手動でプロンプトを作成して合成データを生成する方法を探ります。

### 基本的なプロンプト

基本的な例から始めましょう。[HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)モデルを`distilabel`ライブラリの`transformers`統合を使用してロードします。`TextGeneration`クラスを使用して合成`プロンプト`を生成し、それを使用して`completion`を生成します。

次に、`distilabel`ライブラリを使用してモデルをロードします。

```python
from distilabel.llms import TransformersLLM
from distilabel.steps.tasks import TextGeneration

llm = TransformersLLM(model="HuggingFaceTB/SmolLM2-1.7B-Instruct")
gen = TextGeneration(llm=llm)
gen.load()
```

!!! note
    Distilabelは`llm`をメモリにロードするため、ノートブックで作業する場合は、メモリの問題を避けるために使用後に`gen.unload()`する必要があります。

次に、`llm`を使用してインストラクションチューニングのための`プロンプト`を生成します。

```python
next(gen.process([{"instruction": "Hugging Faceの小規模AIモデルに関するSmol-Courseについての質問を生成してください。"}]))
# Smol-Courseの目的は何ですか？
```

最後に、同じ`プロンプト`を入力として使用して`completion`を生成します。

```python
next(gen.process([{"instruction": "Smol-Courseの目的は何ですか？"}]))
# Smol-Courseはコンピュータサイエンスの概念を学ぶためのプラットフォームです。
```

素晴らしい！合成`プロンプト`と対応する`completion`を生成できました。このシンプルなアプローチをスケールアップして、より多くのデータを生成することができますが、データの品質はそれほど高くなく、コースやドメインのニュアンスを考慮していません。さらに、現在のコードを再実行すると、データがそれほど多様でないことがわかります。幸いなことに、この問題を解決する方法があります。

### SelfInstruct

SelfInstructは、シードデータセットに基づいて新しいインストラクションを生成するプロンプトです。このシードデータは単一のインストラクションやコンテキストの一部である場合があります。プロセスは、初期のシードデータのプールから始まります。言語モデルは、インコンテキスト学習を使用してこのシードデータに基づいて新しいインストラクションを生成するようにプロンプトされます。このプロンプトは[distilabelで実装されています](https://github.com/argilla-io/distilabel/blob/main/src/distilabel/steps/tasks/templates/self-instruct.jinja2)が、簡略化されたバージョンは以下の通りです：

```
# タスクの説明
与えられたAIアプリケーションが受け取ることができる{{ num_instructions }}のユーザークエリを開発してください。モデルのテキスト能力内で動詞と言語構造の多様性を強調してください。

# コンテキスト
{{ input }}

# 出力
```

これを使用するには、`llm`を[SelfInstructクラス](https://distilabel.argilla.io/dev/components-gallery/tasks/selfinstruct/)に渡す必要があります。[プロンプトからデータセクション](#prompt-to-data)のテキストをコンテキストとして使用し、新しいインストラクションを生成してみましょう。

```python
from distilabel.steps.tasks import SelfInstruct

self_instruct = SelfInstruct(llm=llm)
self_instruct.load()

context = "<prompt_to_data_section>"

next(self_instruct.process([{"input": text}]))["instructions"][0]
# 手動プロンプトを使用して合成データを生成するプロセスは何ですか？
```

生成されたインストラクションはすでにかなり良くなっており、実際のコンテンツやドメインに適しています。しかし、プロンプトを進化させることでさらに良くすることができます。

### EvolInstruct

EvolInstructは、入力インストラクションを進化させて、同じインストラクションのより良いバージョンにするプロンプト技術です。このより良いバージョンは、一連の基準に従って定義され、元のインストラクションに制約、深化、具体化、推論、または複雑化を追加します。このプロセスは、元のインストラクションのさまざまな進化を作成するために複数回繰り返すことができ、理想的には元のインストラクションのより良いバージョンに導きます。このプロンプトは[distilabelで実装されています](https://github.com/argilla-io/distilabel/tree/main/src/distilabel/steps/tasks/evol_instruct)が、簡略化されたバージョンは以下の通りです：

```
プロンプトリライターとして行動してください。
与えられたプロンプトを、より複雑なバージョンに書き換えてください。
以下の基準に基づいてプロンプトを複雑化してください：
{{ criteria }}

# プロンプト
{{ input }}

# 出力
```

これを使用するには、`llm`を[EvolInstructクラス](https://distilabel.argilla.io/dev/components-gallery/tasks/evolinstruct/)に渡す必要があります。[SelfInstructセクション](#selfinstruct)の合成プロンプトを入力として使用し、それをより良いバージョンに進化させてみましょう。この例では、1世代だけ進化させます。

```python
from distilabel.steps.tasks import EvolInstruct

evol_instruct = EvolInstruct(llm=llm, num_evolutions=1)
evol_instruct.load()

text = "手動プロンプトを使用して合成データを生成するプロセスは何ですか？"

next(evol_instruct.process([{"instruction": text}]))
# 手動プロンプトを使用して合成データを生成するプロセスは何ですか？
# そして、人工知能システムであるGPT4が機械学習アルゴリズムを使用して入力データを合成データに変換する方法は？
```

インストラクションはより複雑になりましたが、元の意味を失っています。したがって、進化させることは両刃の剣であり、生成するデータの品質に注意する必要があります。

### Magpie

Magpieは、言語モデルの自己回帰的要因と、インストラクションチューニングプロセス中に使用されていた[チャットテンプレート](../1_instruction_tuning/chat_templates.md)に依存する技術です。覚えているかもしれませんが、チャットテンプレートは、システム、ユーザー、アシスタントの役割を明確に示す形式で会話を構造化します。インストラクションチューニングフェーズ中に、言語モデルはこの形式を再現するように最適化されており、Magpieはそれを利用します。チャットテンプレートに基づいた事前クエリプロンプトから始めますが、ユーザーメッセージインジケーター、例：<|im_end|>ユーザー\nの前で停止し、その後、言語モデルを使用してユーザープロンプトを生成し、アシスタントインジケーター、例：<|im_end|>まで生成します。このアプローチにより、多くのデータを非常に効率的に生成でき、マルチターンの会話にもスケールアップできます。この生成されたデータは、使用されたモデルのインストラクションチューニングフェーズのトレーニングデータを再現すると仮定されています。

このシナリオでは、プロンプトテンプレートはチャットテンプレート形式に基づいているため、モデルごとに異なります。しかし、プロセスをステップバイステップで簡略化して説明できます。

```bash
# ステップ1：事前クエリプロンプトを提供する
<|im_end|>ユーザー\n

# ステップ2：言語モデルがユーザープロンプトを生成する
<|im_end|>ユーザー\n
Smol-Courseの目的は何ですか？

# ステップ3：生成を停止する
<|im_end|>
```

distilabelで使用するには、`llm`を[Magpieクラス](https://distilabel.argilla.io/dev/components-gallery/tasks/magpie/)に渡す必要があります。

```python
from distilabel.steps.tasks import Magpie

magpie = Magpie(llm=llm)
magpie.load()

next(magpie.process([{"system_prompt": "あなたは役立つアシスタントです。"}]))
# [{
#   "role": "user",
#   "content": "トップ3の大学のリストを提供できますか？"
# },
# {
#   "role": "assistant",
#   "content": "トップ3の大学は：MIT、イェール、スタンフォードです。"
# }]
```

すぐに`プロンプト`と`completion`を含むデータセットが得られます。ドメインに特化したパフォーマンスを向上させるために、`system_prompt`に追加のコンテキストを挿入できます。LLMが特定のドメインデータを生成するために、システムプロンプトでユーザーのクエリがどのようなものかを説明することが役立ちます。これは、ユーザープロンプトを生成する前の事前クエリプロンプトで使用され、LLMがそのドメインのユーザークエリを生成するようにバイアスをかけます。

```
あなたは数学の問題を解決するためにユーザーを支援するAIアシスタントです。
```

システムプロンプトに追加のコンテキストを渡すことは、一般的に言語モデルが最適化されていないため、カスタマイズには他の技術ほど効果的ではないことが多いです。

### プロンプトからパイプラインへ

これまで見てきたクラスはすべて、パイプラインで使用できるスタンドアロンのクラスです。これは良いスタートですが、`Pipeline`クラスを使用してデータセットを生成することでさらに良くすることができます。`TextGeneration`ステップを使用してインストラクションチューニングのための合成データセットを生成します。パイプラインは、データをロードするための`LoadDataFromDicts`ステップ、`プロンプト`を生成するための`TextGeneration`ステップ、およびそのプロンプトの`completion`を生成するためのステップで構成されます。ステップを接続し、`>>`演算子を使用してデータをパイプラインに流します。distilabelの[ドキュメント](https://distilabel.argilla.io/dev/components-gallery/tasks/textgeneration/#input-output-columns)には、ステップの入力および出力列に関する情報が記載されています。データがパイプラインを正しく流れるようにするために、`output_mappings`パラメータを使用して出力列を次のステップの入力列にマッピングします。

```python
from distilabel.llms import TransformersLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration

with Pipeline() as pipeline:
    data = LoadDataFromDicts(data=[{"instruction": "Hugging Faceの小規模AIモデルに関するSmol-Courseについての短い質問を生成してください。"}])
    llm = TransformersLLM(model="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    gen_a = TextGeneration(llm=llm, output_mappings={"generation": "instruction"})
    gen_b = TextGeneration(llm=llm, output_mappings={"generation": "response"})
    data >> gen_a >> gen_b

if __name__ == "__main__":
    distiset = pipeline.run(use_cache=False)
    print(distiset["default"]["train"][0])
# [{
#   "instruction": "Smol-Courseの目的は何ですか？",
#   "response": "Smol-Courseはコンピュータサイエンスの概念を学ぶためのプラットフォームです。"
# }]
```

このパイプラインの背後には多くのクールな機能があります。生成結果を自動的にキャッシュするため、生成ステップを再実行する必要がありません。フォールトトレランスが組み込まれており、生成ステップが失敗してもパイプラインは実行を続けます。また、すべての生成ステップを並行して実行するため、生成が高速です。`draw`メソッドを使用してパイプラインを視覚化することもできます。ここでは、データがパイプラインをどのように流れ、`output_mappings`が出力列を次のステップの入力列にどのようにマッピングするかを確認できます。

![Pipeline](./images/pipeline.png)

## ベストプラクティス

- 多様なシードデータを確保して、さまざまなシナリオをカバーする
- 生成されたデータが多様で高品質であることを定期的に評価する
- データの品質を向上させるために（システム）プロンプトを繰り返し改善する

## 次のステップ

👨🏽‍💻 コード - [演習ノートブック](./notebooks/instruction_sft_dataset.ipynb)でインストラクションチューニングのためのデータセットを生成する
🧑‍🏫 学ぶ - [preference datasetsの生成](./preference_datasets.md)について学ぶ

## 参考文献

- [Distilabel Documentation](https://distilabel.argilla.io/latest/)
- [Self-instruct](https://arxiv.org/abs/2212.10560)
- [Evol-Instruct](https://arxiv.org/abs/2304.12244)
- [Magpie](https://arxiv.org/abs/2406.08464)
