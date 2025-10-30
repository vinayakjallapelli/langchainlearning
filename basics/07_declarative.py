import os
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

os.getenv("OPENAI_API_KEY")


template = ChatPromptTemplate.from_messages([
    ('system','You are a helpful assistant'),
    ('human', '{question}')
])

model = ChatOpenAI()

chatbot = template | model

print(chatbot.invoke({"question":"Which model providers offer LLMs?"}))