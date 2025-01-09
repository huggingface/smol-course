# Basic RAG

## **Overview**
The Retrieval-Augmented Generation (RAG) pipeline is a powerful framework for combining **retrieval** and **generation** to provide factual, context-aware responses. The pipeline uses a knowledge base of documents, which is indexed and retrieved efficiently to generate answers grounded in the stored information. 

RAG pipelines are particularly useful for tasks requiring factual consistency, such as question answering, research assistance, and domain-specific knowledge systems. In this setup, we leverage **Haystack**, a modular framework for building RAG pipelines with relatively low complexity compared to other popular frameworks like **LangChain** and **Llama-Index**.


## **Why Haystack?**
**Haystack** is chosen over more popular frameworks like LangChain and Llama-Index because of the following advantages:

1. **Less Abstraction**: Haystack provides a straightforward implementation with clear, modular components for indexing and retrieval.
2. **Lightweight**: It avoids unnecessary overhead and keeps the setup simpler for use cases that don't require extensive customizations or orchestration.
3. **Transparency**: Each step in the pipeline is explicit, making it easier to debug, modify, or extend.
4. **Feature-rich**: Haystack includes out-of-the-box support for document cleaning, splitting, embedding, and indexing, alongside retrieval and generation.
5. **Broad Compatibility**: Works seamlessly with pre-trained models from **Hugging Face**, **Sentence Transformers**, and other embedding libraries.


## **Indexing Pipeline**
The indexing pipeline is the foundation of the RAG framework. It preprocesses documents, converts them into embeddings, and stores them in a searchable format for efficient retrieval.

**Steps in the Indexing Pipeline**
1. **Document Collection**: Collect raw documents (e.g., Wikipedia pages) and parse them into structured data.  
2. **Document Cleaning**: Remove irrelevant or noisy text using the **DocumentCleaner**.  
3. **Document Splitting**: Divide documents into smaller chunks (paragraphs or sentences) using the **DocumentSplitter**.  
4. **Embedding Generation**: Generate embeddings (vector representations) of each chunk using **SentenceTransformersDocumentEmbedder**.  
5. **Document Indexing**: Store the processed documents and their embeddings in an **InMemoryDocumentStore**.

```plaintext
[Raw Documents]
       |
       v
[Document Cleaner] -- Removes noise
       |
       v
[Document Splitter] -- Splits into chunks
       |
       v
[Document Embedder] -- Converts chunks into vectors
       |
       v
[Document Store] -- Stores vectors for retrieval
```

At the end of this pipeline, all documents are preprocessed, split into manageable pieces, and embedded into a format that supports fast semantic search.

## **Retrieve + Generate Pipeline**
The generation pipeline combines the retrieval step with a text generation model to answer user queries effectively.

**Steps in the Generation Pipeline**
1. **Query Embedding**: The user’s query is converted into a vector using a **SentenceTransformersTextEmbedder**.  
2. **Document Retrieval**: The query vector is compared to document vectors in the **DocumentStore**, and the most relevant documents are retrieved based on similarity.  
3. **Prompt Construction**: The retrieved documents and user query are formatted into a prompt using the **PromptBuilder**.  
4. **Response Generation**: The prompt is passed to a generative model (e.g., **SmolLM2**) to produce the final answer.

```plaintext
[User Query]
       |
       v
[Query Embedder] -- Converts query into a vector
       |
       v
[Document Retriever] -- Finds top-k relevant documents
       |
       v
[Prompt Builder] -- Combines query + documents into a prompt
       |
       v
[Text Generator] -- Produces answer based on the prompt
```

This pipeline ensures that the generated response is informed by the retrieved context, making it more factual and relevant to the query.

## **Evaluation**