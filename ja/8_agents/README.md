# エージェント

AIエージェントは、ユーザーの要求を理解し、それらをステップに分解し、タスクを達成するためにアクションを実行できる自律システムです。これらは、言語モデルとツールおよび外部関数を組み合わせて環境と対話します。このモジュールでは、軽量フレームワークを提供する[`smolagents`](https://github.com/huggingface/smolagents)ライブラリを使用して、効果的なエージェントを構築する方法を説明します。

## モジュール概要

効果的なエージェントを構築するには、3つの重要なコンポーネントを理解する必要があります。まず、エージェントがさまざまなソースから関連情報にアクセスして使用できるようにする取得機能です。次に、エージェントが環境で具体的なアクションを実行できるようにする関数呼び出しです。最後に、コード操作などの専門的なタスクにエージェントを装備するためのドメイン固有の知識とツールです。

## コンテンツ

### 1️⃣ [取得エージェント](./retrieval_agents.md)

取得エージェントは、モデルと知識ベースを組み合わせます。これらのエージェントは、ベクトルストアを使用して効率的に取得し、RAG（取得強化生成）パターンを実装することで、複数のソースから情報を検索および合成できます。これらは、メモリシステムを通じて会話のコンテキストを維持しながら、Web検索とカスタム知識ベースを組み合わせるのに優れています。このモジュールでは、堅牢な情報取得のためのフォールバックメカニズムを含む実装戦略を説明します。

### 2️⃣ [コードエージェント](./code_agents.md)

コードエージェントは、ソフトウェア開発タスク用に設計された専門の自律システムです。これらのエージェントは、コードの分析と生成、自動リファクタリングの実行、および開発ツールとの統合に優れています。このモジュールでは、プログラミング言語を理解し、ビルドシステムと連携し、バージョン管理と対話しながら高いコード品質基準を維持できるコード中心のエージェントを構築するためのベストプラクティスを説明します。

### 3️⃣ [カスタム関数](./custom_functions.md)

カスタム関数エージェントは、専門の関数呼び出しを通じて基本的なAI機能を拡張します。このモジュールでは、アプリケーションのロジックと直接統合するモジュール式で拡張可能な関数インターフェースを設計する方法を探ります。信頼性の高い関数駆動のワークフローを作成しながら、適切な検証とエラーハンドリングを実装する方法を学びます。焦点は、エージェントが外部ツールやサービスと予測可能に対話できるシンプルなシステムの構築にあります。

### 演習ノートブック

| タイトル | 説明 | 演習 | リンク | Colab |
|-------|-------------|----------|------|-------|
| 研究エージェントの構築 | 取得とカスタム関数を使用して研究タスクを実行できるエージェントを作成 | 🐢 シンプルなRAGエージェントの構築 <br> 🐕 カスタム検索関数の追加 <br> 🦁 完全な研究アシスタントの作成 | [ノートブック](./notebooks/agents.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/8_agents/notebooks/agents.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

## リソース

- [smolagents Documentation](https://huggingface.co/docs/smolagents) - smolagentsライブラリの公式ドキュメント
- [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) - エージェントアーキテクチャに関する研究論文
- [Agent Guidelines](https://huggingface.co/docs/smolagents/tutorials/building_good_agents) - 信頼性の高いエージェントを構築するためのベストプラクティス
- [LangChain Agents](https://python.langchain.com/docs/how_to/#agents) - エージェント実装の追加例
- [Function Calling Guide](https://platform.openai.com/docs/guides/function-calling) - LLMでの関数呼び出しの理解
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/) - 効果的なRAGの実装ガイド
