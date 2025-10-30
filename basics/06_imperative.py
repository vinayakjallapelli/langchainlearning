import os
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain

os.getenv("OPENAI_API_KEY")


template = ChatPromptTemplate.from_messages([
    ('system','You are a helpful assistant'),
    ('human', '{question}')
])

model = ChatOpenAI()


@chain
def chatbot(values):
    prompt = template.invoke(values)
    return model.invoke(prompt)


print(chatbot.invoke({"question":"Which model providers offer LLMs?"}))