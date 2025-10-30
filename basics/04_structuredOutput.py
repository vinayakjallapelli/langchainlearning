import os

from langchain_openai import ChatOpenAI
from pydantic import BaseModel


os.getenv("OPENAI_API_KEY")


class AnswerWithJustification(BaseModel):
    answer: str
    '''the answer to user's question'''
    justification: str
    '''justification to the answer'''


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
structured_llm = llm.with_structured_output(AnswerWithJustification)


print(structured_llm.invoke("""What weighs more, a pound of bricks or a pound of feathers"""))