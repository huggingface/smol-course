# **Basic RAG (Retrieval-Augmented Generation)**

## **Overview**

Retrieval-Augmented Generation (RAG) is a powerful framework for combining **retrieval** (fetching relevant context) with **generation** (producing coherent and contextually accurate responses) to create intelligent, factual, and context-aware systems. 

Most RAG pipeline leverages **retrieval** to access external knowledge and **generation** to produce fluent, natural responses, making it an essential architecture for modern AI systems.

This guide focuses on implementing a basic RAG pipeline using **Haystack**, a lightweight yet feature-rich framework that simplifies the process of building and customizing such systems.


## **Why Haystack?**

Compared to frameworks like LangChain and Llama-Index, Haystack stands out with several advantages:

1. **Minimal Abstraction**: Components like indexing, embedding, and retrieval are modular and transparent, enabling fine-grained control.
2. **Lightweight and Simple**: Focused on core RAG functionalities without excessive abstraction layers, making it more suitable for straightforward use cases.
3. **Debuggable and Extendable**: Each stage in the pipeline is explicit, making debugging easier and enabling customization.
4. **Integrated Features**: Provides robust tools for text preprocessing, embedding generation, and document indexing.
5. **Compatibility**: Works well with pre-trained models (e.g., Hugging Face, Sentence Transformers) and various storage backends.

## **Core Concepts in RAG**

### **Indexing Pipeline**

The **Indexing Pipeline** prepares your knowledge base by preprocessing raw documents, splitting them into manageable chunks, and embedding them into vectors. These vectors are stored in an efficient database to support fast retrieval.

#### **Steps in the Indexing Pipeline**

1. **Document Collection**:
   - Source raw documents from relevant repositories, such as Wikipedia, internal databases, or research articles.
   - Examples: `.txt`, `.pdf`, `.docx`, JSON, or other formats.

2. **Document Cleaning**:
   - Use tools like Haystack's `DocumentCleaner` to remove noise, boilerplate text, and irrelevant sections.
   - Focus on retaining meaningful content.

3. **Document Splitting**:
   - Split large documents into smaller, coherent chunks (e.g., paragraphs or sentences).
   - Use Haystack's `DocumentSplitter` to define chunk size and overlap for better retrieval performance.

4. **Embedding Generation**:
   - Convert text chunks into dense vector representations using pre-trained models (e.g., `SentenceTransformersDocumentEmbedder`).
   - Embeddings capture semantic meaning, enabling similarity-based search.

5. **Document Indexing**:
   - Store embeddings and metadata in a vector database or document store, such as `InMemoryDocumentStore` or `FAISS`.

**Indexing Workflow**

```plaintext
[Raw Documents]
       |
       v
[Document Cleaner] -- Removes noise
       |
       v
[Document Splitter] -- Splits text into chunks
       |
       v
[Document Embedder] -- Converts chunks into vector embeddings
       |
       v
[Document Store] -- Stores embeddings for fast retrieval
```


### **Retrieve + Generate Pipeline**

The **Retrieve + Generate Pipeline** processes user queries by retrieving relevant knowledge and generating context-aware responses using retrieved content.

#### **Steps in the Retrieve + Generate Pipeline**

1. **Query Embedding**:
   - Convert the user query into a dense vector representation using a model like `SentenceTransformersTextEmbedder`.

2. **Document Retrieval**:
   - Perform similarity search in the document store to retrieve the top-k most relevant chunks based on query embedding.

3. **Prompt Construction**:
   - Combine the user query and retrieved documents into a structured prompt for the generative model.
   - Ensure clarity and relevance by organizing context logically.

4. **Response Generation**:
   - Use a text generation model (e.g., GPT-3, SmolLM2) to generate a coherent and factual response based on the constructed prompt.

**Retrieve + Generate Workflow**

```plaintext
[User Query]
       |
       v
[Query Embedder] -- Converts query into vector
       |
       v
[Document Retriever] -- Finds top-k relevant documents
       |
       v
[Prompt Builder] -- Combines query + retrieved documents into a prompt
       |
       v
[Text Generator] -- Produces contextually grounded response
```


## **Evaluation**

To ensure the RAG system performs well, evaluate both retrieval and generation components using appropriate metrics:

- **BLEU** (Bilingual Evaluation Understudy) focuses on **precision** and evaluates how much of the generated text matches reference text n-grams.
- **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) focuses on **recall** and evaluates how much of the reference text's n-grams are captured by the generated text, making it ideal for summarization and text generation tasks.
- **MRR** (Mean Reciprocal Rank) evaluates the effectiveness of an information retrieval system and tasks like question answering by considering the rank of the first relevant result.

Feedback may be used to iteratively improve embeddings, retrieval thresholds, or prompt formatting.

## **Example: Basic RAG System**

1. **Setup Knowledge Base**:
   - Collect documents and preprocess them using the **Indexing Pipeline**.
   
2. **Integrate Query Handling**:
   - Implement the **Retrieve + Generate Pipeline** to handle user inputs.
   
3. **Evaluate and Adjust**:
   - Evaluate the pipeline and monitor retrieval and generation quality. Incorporate feedback for adjustment.

⏩ Try the [Basic RAG Tutorial](./notebooks/naive_rag_haystack_example.ipynb) to implement a Naive RAG pipeline.