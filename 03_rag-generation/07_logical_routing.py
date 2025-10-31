import os
from typing import Literal
from langchain_openai.chat_models import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI()


class RouteQuery(BaseModel):
    """Route a user query to most relevant data source"""

    datasource: Literal["python_docs", "js_docs"] = Field(
        description="""Given a user question, choose which datasource would be most relevant for answering their
         question"""
    )


structured_llm = llm.with_structured_output(RouteQuery)

system_message = """You are an expert at routing a user question to the appropriate data source. Based on the 
    programming language the question is referring to, route it to the relevant data source."""

prompt = ChatPromptTemplate.from_messages([
    ('system', system_message),
    ('human', "{question}")
])

router = prompt | structured_llm

question = """Why doesn't the following code work:
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
    prompt.invoke("french") """

result = router.invoke({"question": question})

print('Selected Datasource: ', result.datasource)


def choose_route(result):
    if "python_docs" in result.datasource.lower():
        ### Logic here
        return "chain for python_docs"
    else:
        ### Logic here
        return "chain for js_docs"


full_chain = router | RunnableLambda(choose_route)

llm_answer = full_chain.invoke(question)
print('LLM Answer: ', llm_answer)
