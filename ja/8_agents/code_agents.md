# コードエージェント

コードエージェントは、分析、生成、リファクタリング、テストなどのコーディングタスクを処理する専門の自律システムです。これらのエージェントは、プログラミング言語、ビルドシステム、バージョン管理に関するドメイン知識を活用して、ソフトウェア開発ワークフローを強化します。

## なぜコードエージェントなのか？

コードエージェントは、反復的なタスクを自動化しながらコード品質を維持することで、開発を加速します。これらは、ボイラープレートコードの生成、体系的なリファクタリングの実行、静的解析を通じた潜在的な問題の特定に優れています。エージェントは、外部のドキュメントやリポジトリにアクセスするための取得機能と、ファイルの作成やテストの実行などの具体的なアクションを実行するための関数呼び出しを組み合わせます。

## コードエージェントの構成要素

コードエージェントは、コード理解のためにファインチューニングされた特殊な言語モデルに基づいて構築されています。これらのモデルは、リンター、フォーマッター、コンパイラーなどの開発ツールで補強され、実世界の環境と相互作用します。検索技術により、エージェントはドキュメントやコード履歴にアクセスし、組織のパターンや標準に合わせることで、コンテキスト認識を維持します。アクション指向の機能により、エージェントは変更のコミットやマージ要求の開始といった具体的なタスクを実行できます。

次の例では、以前に構築した取得エージェントのように、DuckDuckGoを使用してWebを検索できるコードエージェントを作成します。

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())

agent.run("フルスピードのヒョウがポンデザールを走り抜けるのに何秒かかりますか？")
```

次の例では、2つの場所間の移動時間を取得できるコードエージェントを作成します。ここでは、`@tool`デコレーターを使用してツールとして使用できるカスタム関数を定義します。

```python
from smolagents import CodeAgent, HfApiModel, tool

@tool
def get_travel_duration(start_location: str, destination_location: str, departure_time: Optional[int] = None) -> str:
    """2つの場所間の車での移動時間を取得します。
    
    Args:
        start_location: 出発地
        destination_location: 到着地
        departure_time: 出発時間、指定する場合は`datetime.datetime`を提供してください
    """
    import googlemaps # すべてのインポートは関数内に配置され、Hubへの共有を可能にします。
    import os

    gmaps = googlemaps.Client(os.getenv("GMAPS_API_KEY"))

    if departure_time is None:
        from datetime import datetime
        departure_time = datetime(2025, 1, 6, 11, 0)

    directions_result = gmaps.directions(
        start_location,
        destination_location,
        mode="transit",
        departure_time=departure_time
    )
    return directions_result[0]["legs"][0]["duration"]["text"]

agent = CodeAgent(tools=[get_travel_duration], model=HfApiModel(), additional_authorized_imports=["datetime"])

agent.run("パリ周辺でいくつかの場所を訪れる1日の旅行プランを教えてください。都市内でも郊外でも構いませんが、1日で収まるようにしてください。公共交通機関のみを利用します。")

```

これらの例は、コードエージェントでできることの始まりに過ぎません。コードエージェントの構築方法について詳しくは、[smolagentsのドキュメント](https://huggingface.co/docs/smolagents)をご覧ください。

smolagentsは、コードエージェントを構築するための軽量フレームワークを提供し、コア実装は約1,000行のコードで構成されています。このフレームワークは、Pythonコードスニペットを記述および実行するエージェントに特化しており、セキュリティのためにサンドボックス化された実行を提供します。オープンソースおよびプロプライエタリな言語モデルの両方をサポートしており、さまざまな開発環境に適応可能です。

## さらなる学習

- [smolagents Blog](https://huggingface.co/blog/smolagents) - smolagentsとコードインタラクションの紹介
- [smolagents: Building Good Agents](https://huggingface.co/docs/smolagents/tutorials/building_good_agents) - 信頼性の高いエージェントのベストプラクティス
- [Building Effective Agents - Anthropic](https://www.anthropic.com/research/building-effective-agents) - エージェント設計の原則
