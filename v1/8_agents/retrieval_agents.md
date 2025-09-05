# Building Agentic RAG Systems

Agentic RAG (Retrieval Augmented Generation) combines the power of autonomous agents with knowledge retrieval capabilities. While traditional RAG systems simply use an LLM to answer queries based on retrieved information, agentic RAG takes this further by allowing the system to intelligently control its own retrieval and response process.

Traditional RAG has key limitations - it only performs a single retrieval step and relies on direct semantic similarity with the user query, which can miss relevant information. Agentic RAG addresses these challenges by empowering the agent to formulate its own search queries, critique results, and perform multiple retrieval steps as needed.

## Basic Retrieval with DuckDuckGo

Let's start by building a simple agent that can search the web using DuckDuckGo. This agent will be able to answer questions by retrieving relevant information and synthesizing responses.

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

The agent will:
1. Analyze the query to determine what information is needed
2. Use DuckDuckGo to search for relevant content
3. Synthesize the retrieved information into a coherent response
4. Store the interaction in its memory for future reference

## Custom Knowledge Base Tool

For domain-specific applications, we often want to combine web search with our own knowledge base. Let's create a custom tool that can query a vector database of technical documentation.

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

This enhanced agent can:
1. First check the documentation for relevant information
2. Fall back to web search if needed
3. Combine information from both sources
4. Maintain conversation context through memory

## Enhanced Retrieval Capabilities

When building agentic RAG systems, the agent can employ sophisticated strategies like:

1. Query Reformulation - Instead of using the raw user query, the agent can craft optimized search terms that better match the target documents
2. Multi-Step Retrieval - The agent can perform multiple searches, using initial results to inform subsequent queries
3. Source Integration - Information can be combined from multiple sources like web search and local documentation
4. Result Validation - Retrieved content can be analyzed for relevance and accuracy before being included in responses

Effective agentic RAG systems require careful consideration of several key aspects. The agent should select between available tools based on the query type and context. Memory systems help maintain conversation history and avoid repetitive retrievals. Having fallback strategies ensures the system can still provide value even when primary retrieval methods fail. Additionally, implementing validation steps helps ensure the accuracy and relevance of retrieved information.

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

## Next Steps

⏩ Check out the [Code Agents](./code_agents.md) module to learn how to build agents that can manipulate code.
