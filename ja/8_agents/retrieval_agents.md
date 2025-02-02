# エージェントRAGシステムの構築

エージェントRAG（検索拡張生成）は、自律エージェントの力を知識取得機能と組み合わせたものです。従来のRAGシステムは、取得した情報に基づいてクエリに回答するためにLLMを使用しますが、エージェントRAGはシステムが独自の取得および応答プロセスをインテリジェントに制御できるようにします。

従来のRAGには重要な制限があります。それは、単一の取得ステップしか実行せず、ユーザーのクエリと直接的な意味的類似性に依存するため、関連情報を見逃す可能性があることです。エージェントRAGは、エージェントが独自の検索クエリを作成し、結果を批判し、必要に応じて複数の取得ステップを実行できるようにすることで、これらの課題に対処します。

## DuckDuckGoを使用した基本的な取得

まず、DuckDuckGoを使用してWebを検索できるシンプルなエージェントを構築しましょう。このエージェントは、関連情報を取得し、応答を合成することで質問に答えることができます。

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

# 検索ツールを初期化
search_tool = DuckDuckGoSearchTool()

# モデルを初期化
model = HfApiModel()

agent = CodeAgent(
    model = model,
    tools=[search_tool]
)

# 使用例
response = agent.run(
    "核融合エネルギーの最新の進展は何ですか？"
)
print(response)
```

エージェントは次のことを行います：
1. 必要な情報を判断するためにクエリを分析
2. DuckDuckGoを使用して関連コンテンツを検索
3. 取得した情報を合成して一貫した応答を生成
4. 将来の参照のためにインタラクションをメモリに保存

## カスタム知識ベースツール

ドメイン固有のアプリケーションでは、Web検索と独自の知識ベースを組み合わせることがよくあります。カスタムツールを作成して、技術文書のベクトルデータベースをクエリする方法を見てみましょう。

```python
from smolagents import Tool

class RetrieverTool(Tool):
    name = "retriever"
    description = "クエリに最も関連するTransformersのドキュメント部分をセマンティック検索で取得します。"
    inputs = {
        "query": {
            "type": "string",
            "description": "実行するクエリ。ターゲットドキュメントにセマンティックに近い必要があります。質問ではなく肯定形を使用してください。",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(
            docs, k=10
        )

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "検索クエリは文字列である必要があります"

        docs = self.retriever.invoke(
            query,
        )
        return "\n取得したドキュメント:\n" + "".join(
            [
                f"\n\n===== ドキュメント {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )

retriever_tool = RetrieverTool(docs_processed)
```

この強化されたエージェントは次のことができます：
1. 最初にドキュメントを確認して関連情報を探す
2. 必要に応じてWeb検索にフォールバック
3. 両方のソースから情報を組み合わせる
4. メモリを通じて会話のコンテキストを維持

## 強化された取得機能

エージェントRAGシステムを構築する際、エージェントは次のような高度な戦略を採用できます：

1. クエリの再定式化 - 生のユーザークエリを使用する代わりに、ターゲットドキュメントにより適合する検索用語を作成
2. マルチステップ取得 - 初期結果を使用して後続のクエリを情報提供する複数の検索を実行
3. ソース統合 - Web検索やローカルドキュメントなど、複数のソースから情報を組み合わせる
4. 結果の検証 - 取得したコンテンツを分析して関連性と正確性を確認し、応答に含める前に検証

効果的なエージェントRAGシステムには、いくつかの重要な側面を慎重に考慮する必要があります。エージェントは、クエリの種類とコンテキストに基づいて利用可能なツールを選択する必要があります。メモリシステムは会話履歴を維持し、反復取得を回避するのに役立ちます。フォールバック戦略を持つことで、主要な取得方法が失敗した場合でもシステムが価値を提供できるようにします。さらに、検証ステップを実装することで、取得した情報の正確性と関連性を確保するのに役立ちます。

```python
import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")
knowledge_base = knowledge_base.filter(lambda row: row["source"].startswith("huggingface/transformers"))

source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
    for doc in knowledge_base
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)
docs_processed = text_splitter.split_documents(source_docs)
```

## 次のステップ

⏩ [コードエージェント](./code_agents.md)モジュールをチェックして、コードを操作できるエージェントの構築方法を学びましょう。
