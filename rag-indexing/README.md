# RAG Indexing

```mermaid
flowchart LR
    A[Document] --> B[Convert to text\n(Text)]
    B --> C[Split into chunks\n(Chunk 1, Chunk 2, Chunk 3, ...)]
    C --> D[Convert to numbers and store\n(Vector store [0.1, 0.2, 0.5, ...])]
```

## LangChain Document
LangChain provides an abstraction of a document called **Document**, which hosts data, metadata and some index fields. 

LangChain **Document** can be created from multiple sources: Text / Web URL / PDF / Code etc.

LangChain Community provides document loader libraries for converting data from various formats into LangChain Document.

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.langchain.com/")
docs = loader.load()
```


## Generate Embeddings
There are three steps required to generate embeddings for any LangChain Document. 

1. Load the document into LangChain Document
2. Split Document into chunks based on size and overlap requirements
3. Embed chunks using LLM, which generates a vector embedding for each chunk

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings


# load the document
loader = PyPDFLoader("./Learning_LangChain.pdf")
doc = loader.load()

# split the document
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(doc)

# embed the document
model = OpenAIEmbeddings()
embeddings = model.embed_documents([chunk.page_content for chunk in chunks])
```

## Setup Vector Store
We use PGVector store in all examples for storing and retrieveing embeddings from Vector Store.

Use below docker command to spin up PGVector DB on your machine.
```dockerfile
docker run \
--name pgvector-container \
-e POSTGRES_USER=langchain \
-e POSTGRES_PASSWORD=langchain \
-e POSTGRES_DB=langchain \
-p 6024:5432 \
-d pgvector/pgvector:pg16
```
Use below connection string in all later examples.
```postgresql
postgresql+psycopg://langchain:langchain@localhost:6024/langchain
```


## Persist Embeddings in Vector Store
We use PGVector database for storing embeddings. 

PGVector database takes chunks, embedding model and connection string to generate embeddings via provided LLM and persist them in table.
```python
# load the document
loader = PyPDFLoader("./Learning_LangChain.pdf")
doc = loader.load()

# split the document
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(doc)

# embed the document
model = OpenAIEmbeddings()
connection = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'
db = PGVector.from_documents(chunks, model, connection=connection)
```

## Indexing
One problem which arises when using Vector stores, is that documents can change and requires re-generation of embeddings and re-indexing. In this process, same document can get embedded and indexed more than once.

To solve this problem, LangChain offers an API using class called RecordManager. This class keeps track of which documents are indexed and to track duplicates.
```python
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
collection_name = "my_docs"
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
namespace = "my_docs_namespace"

vectorstore = PGVector(
    embeddings=embeddings_model,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

record_manager = SQLRecordManager(
    namespace,
    db_url="postgresql+psycopg://langchain:langchain@localhost:6024/langchain",
)

# Create the schema if it doesn't exist
record_manager.create_schema()

# Create documents
docs = [
    Document(page_content='there are cats in the pond', metadata={"id": 1, "source": "cats.txt"}),
    Document(page_content='ducks are also found in the pond', metadata={"id": 2, "source": "ducks.txt"}),
]

# Index the documents
index_1 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup="incremental", # prevent duplicate documents
    source_id_key="source", # use the source field as the source_id
)
print("Index attempt 1:", index_1)

# second time you attempt to index, it will not add the documents again
index_2 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
)
print("Index attempt 2:", index_2)

# If we mutate a document, the new version will be written and all old
# versions sharing the same source will be deleted.
docs[0].page_content = "I just modified this document!"
index_3 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
)
print("Index attempt 3:", index_3)
```