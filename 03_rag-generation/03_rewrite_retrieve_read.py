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
user_query = """Today I woke up and brushed my teeth, then I sat down to read the news. 
    But then I forgot the food on the cooker. 
    Who are some key figures in the ancient greek history of philosophy?"""

# final llm prompt
prompt = ChatPromptTemplate.from_template("""Answer the question based only on the following context: {context}
    Question: {question} """)

# re-write prompt
rewrite_prompt = ChatPromptTemplate.from_template("""
    Provide a better search query for web search engine to answer the given question, end the queries with ’**’. 
    Question: {x} 
    Answer: """)


# output parser for re-written prompt
def parse_rewriter_output(message):
    return message.content.strip('"').strip('**')


rewriter = rewrite_prompt | llm | parse_rewriter_output

# Runnable to re-write prompt, retrieve documents and read final answer from llm
@chain
def qa_rrr(input):
    new_query = rewriter.invoke(input)
    print('LLM new_query: ', new_query)
    docs = retriever.invoke(new_query)
    final_prompt = prompt.invoke({"context": docs, "question": input})
    answer = llm.invoke(final_prompt)
    return answer


llm_answer = qa_rrr.invoke(user_query)

print('LLM Answer: ', llm_answer)

