import os

from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings

os.getenv("OPENAI_API_KEY")

model = OpenAIEmbeddings()
connection = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'

vectorstore = PGVector(
    embeddings=model,
    collection_name="test",
    connection=connection,
    use_jsonb=True,
)

# create retriever
retriever = vectorstore.as_retriever()

# fetch relevant documents
docs = retriever.invoke("""Who are the key figures in the ancient greek history of philosophy?""")

print(docs)
print(len(docs))

# create retriever with k=2
retriever_limit2 = vectorstore.as_retriever(search_kwargs={"k": 2})

# fetch the 2 most relevant documents
docs = retriever_limit2.invoke("""Who are the key figures in the ancient greek history of philosophy?""")
print(docs)
print(len(docs))

