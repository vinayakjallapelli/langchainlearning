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
rag_fusion_prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant that generates multiple search queries based on a single input query. \n
    Generate multiple search queries related to: {question} \n
    Output (4 queries): """)


# output parser for re-written prompt
def parse_queries_output(message):
    prompts = message.content.split('\n')
    print('LLM Generated Prompts: ', prompts)
    return prompts


query_gen = rag_fusion_prompt | llm | parse_queries_output


def reciprocal_rank_fusion(results: list[list], k=60):
    """reciprocal rank fusion on multiple lists of ranked documents
    and an optional parameter k used in the RRF formula
    """

    # Initialize a dictionary to hold fused scores for each document
    # Documents will be keyed by their contents to ensure uniqueness
    fused_scores = {}
    documents = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list,
        # with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Use the document contents as the key for uniqueness
            doc_str = doc.page_content
            # If the document hasn't been seen yet,
            # - initialize score to 0
            # - save it for later
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
                documents[doc_str] = doc

            # Update the score of the document using the RRF formula:
            # 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order
    # to get the final reranked results
    reranked_doc_strs = sorted(fused_scores, key=lambda d: fused_scores[d], reverse=True)

    # retrieve the corresponding doc for each doc_str
    return [documents[doc_str] for doc_str in reranked_doc_strs]


retrieval_chain = query_gen | retriever.batch | reciprocal_rank_fusion


# Runnable to re-write prompt, retrieve documents and read final answer from llm
@chain
def multi_query_qa(input):
    docs = retrieval_chain.invoke(input)
    final_prompt = prompt.invoke({"context": docs, "question": input})
    answer = llm.invoke(final_prompt)
    return answer


llm_answer = multi_query_qa.invoke(user_query)

print('LLM Answer: ', llm_answer)

