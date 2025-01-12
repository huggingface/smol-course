# Modular (Agentic) RAG

## Why Modular RAG is Needed

**Vanilla RAG** faces two main challenges:

1. **Single retrieval step**: If the retrieved documents are irrelevant, the generated answer will be poor.
2. **Semantic mismatch**: The user’s query might differ in form from the document’s content, making semantic similarity-based retrieval suboptimal.

**Modular RAG** addresses these limitations by introducing an agent that can:

- **Formulate the query** to optimize document retrieval.
- **Critique and re-retrieve** if necessary, improving retrieval accuracy and ensuring better answers.

## Key Components of Modular RAG
### 1. **Module Components**

In a modular RAG system, module, tool, and agent are often used interchangeably, though they represent different levels of abstraction within the system. These components represent different parts of the system that work together to enhance functionality.

- **Search Module**: Expands retrieval by integrating data from various external sources like search engines, tabular data, and knowledge graphs, enhancing the relevance of context during retrieval. 
- **Memory Module**: Stores past interactions (queries and answers) for ongoing context awareness, supporting dynamic tasks and conversations.
- **Custom Function Tool Module**: Executes advanced workflows, such as database queries or system commands, allowing the agent to interact with external systems.
- **Code Module (Agent)**: Specializes in coding tasks like analysis, generation, refactoring, and testing, enabling the agent to handle software development tasks.

### 2. **Other Components**

- **Fusion**: Performs parallel retrieval on original and expanded queries, intelligently reranking and merging results for optimal context.
- **Routing**: Directs the next action based on the query, such as summarization or searching specific databases, ensuring appropriate responses.
- **Orchestration Agent**: Coordinates the flow of information between modules, optimizing the efficiency and effectiveness of the overall RAG system.

### Resources

https://huggingface.co/docs/smolagents/examples/rag