from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.prompts import PromptTemplate
from langchain_openai.llms import OpenAI
from langchain_core.runnables import chain

connection='postgresql+psycopg://langchain:langchain@localhost:6024/langchain'
embedding_model = OpenAIEmbeddings()
collection_name = 'test'
user_query = 'Who are the key figures in the ancient greek history of philosophy?'
template = PromptTemplate.from_template("""
Answer the question based only on the following context:
{context}
Question: {question} """)

vectorDBStore = PGVector(
    embeddings=embedding_model,
    connection=connection,
    collection_name=collection_name,
    use_jsonb=True
)

# create a retriever, which performs embedding on user query and searches vector store for embeddings
retriever = vectorDBStore.as_retriever()
llm = OpenAI()


@chain
def chat(input):
    # fetch documents from vector store
    vectorDocs = retriever.invoke(input)
    prompt = template.invoke({"context": vectorDocs, "question": input})
    answer = llm.invoke(prompt)
    return {"answer": answer, "docs": vectorDocs}


chat_output = chat.invoke(user_query)

print(chat_output)







