import os

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_postgres import PGVector
from langchain_core.runnables import chain
from langchain_openai import OpenAIEmbeddings

# Initialization
os.getenv("OPENAI_API_KEY")
vector_db_connection = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'
vector_db_collection = 'test'
llm = ChatOpenAI()
llm_embeddings = OpenAIEmbeddings()
vector_db = PGVector(
    connection=vector_db_connection,
    embeddings=llm_embeddings,
    collection_name=vector_db_collection,
    use_jsonb=True
)

# vector db retriever
retriever = vector_db.as_retriever()

# user query
user_query = """Who are some key figures in the ancient greek history of philosophy?"""

# final llm prompt
prompt = ChatPromptTemplate.from_template("""Answer the question based only on the following context: {context}
    Question: {question} """)

# re-write prompt
perspectives_prompt = ChatPromptTemplate.from_template("""
    You are an AI language model assistant. Your task is to generate five different versions of the given user question
     to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, 
     your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide 
     these alternative questions separated by newlines. Original question: {question} """)


# output parser for re-written prompt
def parse_queries_output(message):
    prompts = message.content.split('\n')
    print('LLM Generated Prompts: ', prompts)
    return prompts


query_gen = perspectives_prompt | llm | parse_queries_output


def get_unique_union(document_lists):
    # Flatten list of lists, and dedupe them
    deduped_docs = {
        doc.page_content: doc
        for sublist in document_lists for doc in sublist
    }
    # return a flat list of unique docs
    return list(deduped_docs.values())


retrieval_chain = query_gen | retriever.batch | get_unique_union


# Runnable to re-write prompt, retrieve documents and read final answer from llm
@chain
def multi_query_qa(input):
    docs = retrieval_chain.invoke(input)
    final_prompt = prompt.invoke({"context": docs, "question": input})
    answer = llm.invoke(final_prompt)
    return answer


llm_answer = multi_query_qa.invoke(user_query)

print('LLM Answer: ', llm_answer)

