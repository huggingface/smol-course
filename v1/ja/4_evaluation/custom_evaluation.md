# カスタムドメイン評価

標準ベンチマークは貴重な洞察を提供しますが、多くのアプリケーションでは特定のドメインやユースケースに合わせた評価アプローチが必要です。このガイドでは、ターゲットドメインでモデルの性能を正確に評価するためのカスタム評価パイプラインを作成する方法を説明します。

## 評価戦略の設計

成功するカスタム評価戦略は、明確な目標から始まります。ドメインでモデルが示すべき具体的な能力を考慮してください。これには、技術的な知識、推論パターン、ドメイン固有の形式が含まれるかもしれません。これらの要件を慎重に文書化してください。これがタスク設計とメトリック選択の両方を導きます。

評価は、標準的なユースケースとエッジケースの両方をテストする必要があります。例えば、医療ドメインでは、一般的な診断シナリオと稀な状態の両方を評価するかもしれません。金融アプリケーションでは、通常の取引と複数の通貨や特別な条件を含む複雑なエッジケースの両方をテストするかもしれません。

## LightEvalを使用した実装

LightEvalは、カスタム評価を実装するための柔軟なフレームワークを提供します。カスタムタスクを作成する方法は次のとおりです：

```python
from lighteval.tasks import Task, Doc
from lighteval.metrics import SampleLevelMetric, MetricCategory, MetricUseCase

class CustomEvalTask(Task):
    def __init__(self):
        super().__init__(
            name="custom_task",
            version="0.0.1",
            metrics=["accuracy", "f1"],  # 選択したメトリック
            description="カスタム評価タスクの説明"
        )
    
    def get_prompt(self, sample):
        # 入力をプロンプトにフォーマット
        return f"質問: {sample['question']}\n回答:"
    
    def process_response(self, response, ref):
        # モデルの出力を処理し、参照と比較
        return response.strip() == ref.strip()
```

## カスタムメトリック

ドメイン固有のタスクには、専門的なメトリックが必要なことがよくあります。LightEvalは、ドメインに関連する性能の側面を捉えるカスタムメトリックを作成するための柔軟なフレームワークを提供します：

```python
from aenum import extend_enum
from lighteval.metrics import Metrics, SampleLevelMetric, SampleLevelMetricGrouping
import numpy as np

# サンプルレベルのメトリック関数を定義
def custom_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> dict:
    """サンプルごとに複数のスコアを返す例のメトリック"""
    response = predictions[0]
    return {
        "accuracy": response == formatted_doc.choices[formatted_doc.gold_index],
        "length_match": len(response) == len(formatted_doc.reference)
    }

# サンプルごとに複数の値を返すメトリックを作成
custom_metric_group = SampleLevelMetricGrouping(
    metric_name=["accuracy", "length_match"],  # サブメトリックの名前
    higher_is_better={  # 各メトリックで高い値が良いかどうか
        "accuracy": True,
        "length_match": True
    },
    category=MetricCategory.CUSTOM,
    use_case=MetricUseCase.SCORING,
    sample_level_fn=custom_metric,
    corpus_level_fn={  # 各メトリックを集計する方法
        "accuracy": np.mean,
        "length_match": np.mean
    }
)

# LightEvalにメトリックを登録
extend_enum(Metrics, "custom_metric_name", custom_metric_group)
```

サンプルごとに1つのメトリック値のみが必要な場合の簡単なケース：

```python
def simple_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> bool:
    """サンプルごとに単一のスコアを返す例のメトリック"""
    response = predictions[0]
    return response == formatted_doc.choices[formatted_doc.gold_index]

simple_metric_obj = SampleLevelMetric(
    metric_name="simple_accuracy",
    higher_is_better=True,
    category=MetricCategory.CUSTOM,
    use_case=MetricUseCase.SCORING,
    sample_level_fn=simple_metric,
    corpus_level_fn=np.mean  # サンプル全体で集計する方法
)

extend_enum(Metrics, "simple_metric", simple_metric_obj)
```

カスタムメトリックを評価タスクで参照することで、評価タスクで自動的に計算され、指定された関数に従って集計されます。

より複雑なメトリックの場合、次のことを検討してください：
- フォーマットされたドキュメントのメタデータを使用してスコアを重み付けまたは調整
- コーパスレベルの統計のためのカスタム集計関数の実装
- メトリック入力の検証チェックの追加
- エッジケースと期待される動作の文書化

これらの概念を実装する完全な例については、[ドメイン評価プロジェクト](./project/README.md)を参照してください。

## データセットの作成

高品質の評価には、慎重にキュレーションされたデータセットが必要です。データセット作成のアプローチを次に示します：

1. 専門家のアノテーション：ドメインの専門家と協力して評価例を作成および検証します。[Argilla](https://github.com/argilla-io/argilla)のようなツールを使用すると、このプロセスがより効率的になります。

2. 実世界のデータ：実際の使用データを収集し、匿名化して、実際の展開シナリオを反映します。

3. 合成生成：LLMを使用して初期例を生成し、専門家がそれを検証および洗練します。

## ベストプラクティス

- 評価方法論を徹底的に文書化し、仮定や制限を含める
- ドメインのさまざまな側面をカバーする多様なテストケースを含める
- 自動メトリックと人間の評価の両方を適用する
- 評価データセットとコードをバージョン管理する
- 新しいエッジケースや要件を発見するたびに評価スイートを定期的に更新する

## 参考文献

- [LightEvalカスタムタスクガイド](https://github.com/huggingface/lighteval/wiki/Adding-a-Custom-Task)
- [LightEvalカスタムメトリック](https://github.com/huggingface/lighteval/wiki/Adding-a-New-Metric)
- データセットアノテーションのための[Argillaドキュメント](https://docs.argilla.io)
- 一般的な評価原則のための[評価ガイドブック](https://github.com/huggingface/evaluation-guidebook)

# 次のステップ

⏩ これらの概念を実装する完全な例については、[ドメイン評価プロジェクト](./project/README.md)を参照してください。
