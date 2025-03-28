# 构建自主性检索增强生成（RAG）系统

自主性检索增强生成（RAG 或 Retrieval Augmented Generation）将有自主性的智能体的能力与知识检索能力相结合。传统的 RAG 系统只是利用大语言模型（LLM），基于检索到的信息来回答询问。而自主性 RAG 更进一步，允许系统智能地控制自身的检索和回答过程。

传统 RAG 存在关键局限 —— 它仅执行单一的检索步骤，且依赖与用户询问的直接语义相似性，这可能会遗漏相关信息。自主性 RAG 通过让智能体能够自行制定搜索查询、评估结果，并根据需要执行多个检索步骤，来应对这些挑战。

## 使用 DuckDuckGo 实现基本检索

我们首先构建一个简单的智能体，它可以用 DuckDuckGo 去搜索网页。这个智能体可以通过检索相关信息并整合，来回答问题。

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

# Initialize the search tool
search_tool = DuckDuckGoSearchTool()

# Initialize the model
model = HfApiModel()

agent = CodeAgent(
    model = model,
    tools=[search_tool]
)

# Example usage
response = agent.run(
    "What are the latest developments in fusion energy?"
)
print(response)
```

这个智能体背后所做的事情是：
1. 分析这个问题，确定需要哪些参考信息
2. 使用 DuckDuckGo 这个搜索引擎寻找相关内容
3. 整合检索到的信息，形成条理清晰的回答
4. 在内存中保存本次交互信息，以供将来参考

## 自定义知识库

针对特定领域的应用，我们通常希望在网络搜索之外再加入特有的知识库作为参考。接下来我们就创建一个自定义的工具，用以从一个技术文档的向量数据库中进行查询。

```python
from smolagents import Tool

class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve the parts of transformers documentation that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(
            docs, k=10
        )

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.retriever.invoke(
            query,
        )
        return "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )

retriever_tool = RetrieverTool(docs_processed)
```

这个增强版的智能体背后所做的事情是：
1. 首先检查文档，看有没有相关信息First check the documentation for relevant information
2. 如有需要，也进行网页搜索
3. 集成两个信息源的信息
4. 在内存中保存对话的上下文

## 增强检索能力

当构建自主性检索增强生成（RAG）系统时，智能体还可以借鉴以下复杂策略：

1. 查询重构：智能体并非直接使用用户的原始查询，而是精心设计优化后的搜索词，使其与目标文档更匹配。
2. 多步检索：智能体可以执行多次搜索，依据初始检索结果来调整后续的查询。
3. 来源整合：整合来自多个渠道的信息，如网页搜索和本地文档。
4. 结果验证：在将检索到的内容纳入回复之前，会对其相关性和准确性进行分析。

要构建有效的自主性检索增强生成（RAG）系统，需要仔细考量几个关键方面。智能体应依据查询类型和上下文，在可用工具中做出选择。记忆系统有助于保存对话历史，避免重复检索。制定回退策略可确保即便首选的检索方法失效，系统仍能发挥作用。此外，实施验证步骤有助于保证检索信息的准确性和相关性。


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

## 接下来

⏩ 学习[代码智能体](./code_agents_cn.md)部分，了解如何构建一个可以处理代码任务的智能体。
