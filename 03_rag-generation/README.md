# RAG - Retrieval and Generation

## Key Stages of RAG
<img src="images/rag_stages.png" alt="Preview" width="700" height="300">

## Retrieval Process
<img src="images/retrieval_process.png" alt="Preview" width="400" height="400">

## Generation Process
<img src="images/generation_process.png" alt="Preview" width="400" height="400">


## Building Production Grade RAG System
A production grade RAG system must solve for below questions.
- How do we handle the variability in the quality of a user’s input?  _(see Query Transformation)_
- How do we route queries to retrieve relevant data from a variety of data sources? _(see Query Routing)_
- How do we transform natural language to the query language of the target data source? _(see Query Construction)_
- How do we optimize our indexing process, i.e., embedding, text splitting?

Strategies to Optimize RAG System
<img src="images/rag_optimization.png" alt="Preview" width="800" height="400">

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


