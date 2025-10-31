# RAG - Retrieval and Generation

```mermaid
flowchart LR
    Q[Question] --> I[Indexing]
    subgraph I[Indexing]
        D[Documents] --> IDX[Index]
    end
    I --> R[Retrieval \n Relevant document]
    R --> G[Generation \n Content window]
    G --> A[Answer]
```

## Retrieval Process
<img src="images/retrieval_process.png" alt="Preview" width="400" height="400">

## Generation Process
<img src="images/generation_process.png" alt="Preview" width="400" height="400">


## Building Production Grade RAG System
A production grade RAG system must solve for below questions.
- How do we handle the variability in the quality of a user’s input?  _(see Query Transformation)_
- How do we route queries to retrieve relevant data from a variety of data sources?
- How do we transform natural language to the query language of the target data
source?
- How do we optimize our indexing process, i.e., embedding, text splitting?


### Query Transformation: Rewrite-Retrieve-Read 
<img src="images/rewrite-retrieve-read.png" alt="Preview" width="800" height="400">

### Query Transformation: Multi-Query Retrieval
<img src="images/multi_query_retrieval.png" alt="Preview" width="600" height="300">

### Query Transformation: Hypothetical Document Embeddings
<img src="images/hypothetical_document_embedding.png" alt="Preview" width="600" height="300">

### Query Routing: Logical
<img src="images/logical_routing.png" alt="Preview" width="600" height="300">

### Query Routing: Semantic
<img src="images/semantic_routing.png" alt="Preview" width="600" height="300">


