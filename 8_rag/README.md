# Retrieval-Augmented Generation (RAG) Module  

## 1. Overview of RAG  

Retrieval-Augmented Generation (RAG) integrates retrieval mechanisms with generative models to produce context-aware responses by accessing external knowledge dynamically. This architecture is highly modular, allowing for customization and optimization based on task requirements. 

A typical RAG pipeline involves:
- **Retriever**: Fetches relevant documents.
- **Chunker**: Splits documents for efficient retrieval.
- **Generator**: Generates responses based on retrieved context.
- **Orchestrator**: Manages interactions between these components.


## 2. Types of RAG Architectures  

The architecture of a RAG system defines how components are organized and interact. Below are the most common RAG architectures:

### 2.1 Naive RAG  
The simplest architecture where the query is sent to a retriever, and the retrieved chunks are passed directly to the generator for response generation.  
- **Advantages**: Easy to implement, low complexity.  
- **Use Cases**: General-purpose question answering, quick prototypes.  


### 2.2 Retrieve-and-Rerank RAG  
Enhances the naive architecture by incorporating a reranker to prioritize retrieved results.  
- **Components**: Retriever, reranker, generator.  
- **Advantages**: Improves relevance and precision of retrieved context.  
- **Use Cases**: Customer support systems, legal document search.  


### 2.3 Multimodal RAG  
Combines text and visual inputs for tasks that require multimodal reasoning. The retriever handles multimodal datasets, and the generator processes both text and image contexts.  
- **Components**: Multimodal retriever, image encoder, text generator.  
- **Use Cases**: Visual question answering, image captioning, multimodal search.  


### 2.4 Graph RAG  
Leverages graph databases or graph neural networks (GNNs) to model relationships between entities and retrieve structured knowledge.  
- **Components**: Graph retriever, node representation model, generator.  
- **Advantages**: Ideal for reasoning over structured data like knowledge graphs.  
- **Use Cases**: Scientific research, complex entity reasoning, technical documentation.  


### 2.5 Hybrid RAG  
Combines multiple retrieval mechanisms, such as dense vector search and keyword-based search, to ensure robust and diverse retrieval.  
- **Components**: Multi-retriever setup, generator.  
- **Advantages**: Balances precision and recall by integrating complementary retrieval methods.  
- **Use Cases**: Multilingual search, domain-specific retrieval.  


### 2.6 Agentic RAG (Router)  
Uses an agent-based approach to route queries to specialized retrievers or tools based on query type.  
- **Components**: Router agent, multiple retrievers, generator.  
- **Advantages**: Scalable and adaptable to different query types.  
- **Use Cases**: Customer support with diverse query domains.  


### 2.7 Agentic RAG (Multi-Agent RAG)  
Extends the router architecture by involving multiple agents that interact dynamically to solve tasks collaboratively.  
- **Components**: Multiple agents, retrievers, external tools (e.g., Slack, Gmail), generator.  
- **Advantages**: Flexible and supports integration with external systems.  
- **Use Cases**: Workflow automation, enterprise search systems.  


## 3. Components of RAG  

### 3.1 Retriever  
Handles fetching of relevant documents or information from a knowledge base. Popular frameworks include:
- **Vector Databases**: Tools like **Qdrant**, **Weaviate**, and **Pinecone**.
- **Hybrid Retrieval**: Combines keyword and vector search.


### 3.2 Chunker  
Splits documents into smaller chunks for effective retrieval. Tools like **LangChain Text Splitters** are commonly used.


### 3.3 Generator  
Generates responses by processing the retrieved context.  
- **Options**: GPT models, multimodal models (e.g., GPT-4 Vision).  

In this course we will use the Smol-family models as generator.

### 3.4 Orchestrator  
Manages interaction between components and workflows.  
- **Tools**: **LangChain**, **HayStack**.  


## 4. Evaluation  

Evaluation involves measuring retrieval quality and response generation accuracy.  
- **Frameworks**: **RAGAs** for integrated evaluation, **BLEU**, **ROUGE**, and **MRR** for detailed metrics.


## 5. Tools and Frameworks  

| Component       | Tools/Frameworks                      |
|------------------|---------------------------------------|
| Retriever        | **DPR**, **BM25**, **Qdrant**, **Pinecone** |
| Chunker          | **LangChain Text Splitters**          |
| Generator        | **OpenAI GPT-4**, **Hugging Face Transformers** |
| Orchestrator     | **LangChain**, **Hugging Face Agents** |
| Evaluation       | **RAGAs**, **BLEU**, **ROUGE**, **MRR** |


## Exercise Notebooks  

| Title         | Description | Exercise | Link | Colab |
|---------------|-------------|----------|------|-------|
| RAG Basics    | Learn how to construct a RAG pipeline | 🐢 Use a simple retriever and generator<br>🐕 Try vector database integration<br>🦁 Explore hybrid retrieval | [Notebook](./notebooks/rag_basics.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/smol-course/rag_basics.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| RAG Evaluation | Learn how to evaluate retrieval and generation | 🐢 Evaluate a simple RAG system<br>🐕 Use RAGAs for advanced metrics<br>🦁 Analyze retrieval and generation alignment | [Notebook](./notebooks/rag_evaluation.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/smol-course/rag_evaluation.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |


## References  

- [LangChain Documentation](https://docs.langchain.com)  
- [RAGAs Toolkit](https://github.com/ragas-toolkit)  
