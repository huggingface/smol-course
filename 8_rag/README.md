# Retrieval-Augmented Generation (RAG) Module  

## 1. Overview of RAG  

Large Language Models (LLMs) have revolutionized natural language processing and generation. However, their reliance on static training data presents significant challenges. Retrieval-Augmented Generation (RAG) addresses these limitations by combining dynamic retrieval with generative capabilities.  

### Why RAG is Needed

- **Reducing Hallucination**  
RAG grounds responses in external knowledge, ensuring accuracy by dynamically fetching relevant context instead of relying solely on static, memorized information.

- **Real-Time Knowledge**  
Unlike static models, RAG can integrate live data, making it ideal for fast-changing domains like finance, news, and research.

- **Automation and Scalability**  
RAG simplifies complex workflows by dynamically accessing and integrating diverse information sources, enabling scalable applications such as customer support, enterprise search, and workflow automation.

- **Domain-Specific Adaptability**  
With tailored retrievers and modular design, RAG excels in specialized applications such as legal, healthcare, or education, offering flexibility and precision.

- **Cost Efficiency**  
By offloading retrieval tasks to external systems, RAG reduces computational costs and eliminates the need for frequent model fine-tuning.

## 2. Stages of RAG  

Although there are many RAG variants, the main workflow can often be divided into three key stages:

| **Stage**                | **Sub-Components**              |
| ------------------------- | ------------------------------- |
| **Index**                | Ingest, Chunk, Embed, Store     |
| **Retrieve + Generate**     | Retrieve, Generate, Orchestrate |
| **(Optional) Evaluation**| Evaluate responses              |

### 2.1 Index Stage  
The **Index** stage prepares the knowledge base for efficient retrieval by processing and organizing documents. This stage often involves:  
- **Ingesting Documents**: Extracting text from raw documents (e.g., PDFs, HTML).  
- **Chunking**: Splitting documents into smaller, manageable pieces.  
- **Embedding**: Converting chunks into vector representations using embedding models.  
- **Storing**: Indexing the embeddings into a vector database for efficient search.  

- **Tools/Frameworks**:  
  - **Ingesting**: OCR tools, document processors.  
  - **Chunking**: **LangChain Text Splitters**, **HayStack Preprocessors**.  
  - **Embedding**: **OpenAI Embeddings**, **Hugging Face Transformers**.  
  - **Vector Database**: **Qdrant**, **ElasticSearch**, **Pinecone**.  

### 2.2 Retrieve + Generate Stage  
The **Retrieve + Generate** stage retrieves relevant context and generates responses based on it:  

#### **Retrieve**  
Fetches the most relevant chunks of information using:  
- **Retrieval Methods**: Dense vector search, hybrid retrieval (vector + keyword).  
- **Advanced Techniques**: Re-ranking models, custom tools (web search integration, agents, etc.).  

#### **Generate**  
Uses retrieved chunks as input to generate responses:  
- **Generating Methods**: OpenAI, Anthropic, etc. (vision language models for image+text tasks).

#### **(Optional) Orchestrate**  
Manages interactions between retrieval and generation for complex workflows such as in advanced retrival techniques. For example:

- **Prompt Augmentation**: Prepares the final input for the generator by formatting and enriching retrieved chunks (e.g., adding context, query reformulation, or applying templates).  
- **Dynamic Query Refinement**: Iteratively adjusts the query or retrieval parameters based on feedback or partial results to improve the quality of retrieved information.
- **Tool Invocation**: Dynamically calls external tools or APIs (e.g., search engines, databases, or calculators) as part of the response generation process.  

### 2.3. (Optional) Evaluation

Evaluation measures retrieval quality and response generation accuracy.
- **Metrics**: BLEU, ROUGE, MRR.
- **Frameworks**:  Frameworks like RAGAs for evaluation.

 
## 3. The Architecture of RAG Systems  

The architecture of a Retrieval-Augmented Generation (RAG) system determines its capabilities, scalability, and the enhancements it offers beyond the basic Naive RAG approach. Over the years, research and innovation in the RAG space have introduced diverse architectures that optimize various stages of the RAG pipeline.

The distinctions between RAG architectures often emerge from how they handle the **Retrieve** and **Generate** stages. These stages are critical to answering the following questions:  
- **What to retrieve?** Selecting the most relevant chunks or embeddings from the knowledge base.  
- **When to retrieve?** Deciding at what point in the workflow retrieval is required, particularly in iterative or dynamic processes.  
- **How to use the retrieved information?** Determining how retrieved content is incorporated into downstream tasks, such as response generation or query refinement.  

Broadly, they can be categorized into:
- **Naive RAG**:	Most basic pipeline, directly passing retrieved chunks to the generator.
- **Advanced RAG**:	Pipelines which incorperated different techniques to improve the quality of responses with Pre-retrieval and Post-retrieval processes.
- **Modular RAG**:	Pipelines that further enhance functionalities by integrating modules that interatively refine result or dynamically adapt based on task-specific requirements.

In addition, some architectural innovations target the **Index Stage**, introducing new ways to organize and structure knowledge bases:  
- **Graph-Based Knowledge Bases**: Represent data as nodes and edges, enabling richer context and relationship reasoning.  
- **Hierarchical Chunking**: Organizes chunks in a multi-level structure for faster and more contextually aware retrieval.  
- **Adaptive Embedding Updates**: Dynamically adjusts embeddings in response to new data or evolving user needs. 

Following are common RAG architectures:

### 3.1 Naive RAG
A simple architecture where the query is sent to a retriever, and the retrieved chunks are passed directly to the generator for response generation.  
- **Features**: Most rudimentary pipeline with no intermediate steps.  
- **Effect**: Easy to implement with low complexity, suitable for general-purpose question answering and quick prototyping due to the straightforward design.  

### 3.2 Retrieve-and-Rerank RAG
This architecture incorporates a reranker to prioritize retrieved results based on relevance.  
- **Features**: Addition of a reranking model that scores and reorders retrieved documents before passing them to the generator.  
- **Effect**: Improves relevance and precision by filtering out less relevant information, making it more effective for tasks requiring high retrieval accuracy, such as customer support and legal document search.  

### 3.3 Multimodal RAG
Combines text and visual inputs for tasks that require reasoning across multiple data modalities.  
- **Features**: Includes a multimodal retriever, an image encoder, and a text generator capable of handling both text and visual data.  
- **Effect**: Enables tasks such as visual question answering and image captioning by integrating diverse data types. 

### 3.4 Graph RAG
Uses graph databases or graph neural networks (GNNs) to model relationships between entities and retrieve structured knowledge.  
- **Features**: Incorporates a graph retriever and node representation model to leverage entity relationships within a graph structure.  
- **Effect**: Provides enhanced reasoning capabilities over structured data such as knowledge graphs, making it highly effective for tasks like scientific research, complex entity reasoning, and technical documentation.  

### 3.5 Hybrid RAG
Integrates multiple retrieval mechanisms to combine the strengths of different search methods.  
- **Features**: Utilizes both dense vector search and keyword-based retrieval methods in a multi-retriever setup.  
- **Effect**: Balances precision and recall by integrating complementary retrieval approaches, which can be useful for handling diverse or multilingual datasets in domain-specific retrieval tasks.  

### 3.6 Agentic RAG (Router)
Routes queries to specialized retrievers or tools based on their type using an agent-based approach.  
- **Features**: Includes a router agent that dynamically assigns queries to the most appropriate retriever or processing tool.  
- **Effect**: Scales effectively across diverse query types. This allows for adaptability in systems requiring varied query handling, such as customer support across multiple domains.  

### 3.7 Agentic RAG (Multi-Agent RAG)
Expands the router architecture by involving multiple agents that collaborate to solve tasks dynamically.  
- **Features**: Comprises multiple agents, each capable of interacting with retrievers, external tools (e.g., Slack, Gmail), and generators.  
- **Effect**: Enables complex workflows and dynamic task-solving by leveraging agent collaboration. This supports integration with external systems, making it suitable for enterprise-level workflow automation and advanced search systems.  


## Exercise Notebooks  

| Title            | Description | Exercise | Link | Colab |
|-------------------|-------------|----------|------|-------|
| RAG Basics       | Learn how to construct a RAG pipeline | 🐢 Use a simple retriever and generator<br>🐕 Try vector database integration<br>🦁 Explore hybrid retrieval | [Notebook](./notebooks/rag_basics.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/smol-course/rag_basics.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| RAG Evaluation   | Learn how to evaluate retrieval and generation | 🐢 Evaluate a simple RAG system<br>🐕 Use RAGAs for advanced metrics<br>🦁 Analyze retrieval and generation alignment | [Notebook](./notebooks/rag_evaluation.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/smol-course/rag_evaluation.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Agentic RAG      | Learn how to set up and use multi-agent RAG for task automation | 🐢 Create a database query react-agent<br>🐕 Build an agent for solving math/coding problems<br>🦁 Integrate multiple tools for workflow automation | [Notebook](./notebooks/agent_rag.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/smol-course/agent_rag.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |


## Resources

- [Haystack Documentation](https://docs.haystack.deepset.ai/docs/intro)  
- [RAGAs Toolkit](https://github.com/ragas-toolkit)  
- [Youtube: How RAG Turns AI Chatbots Into Something Practical](https://youtu.be/5Y3a61o0jFQ?si=epzQv1UIJe53OoLB)
- https://www.promptingguide.ai/research/rag
