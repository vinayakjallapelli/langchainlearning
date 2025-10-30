import os

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

os.getenv("OPENAI_API_KEY")

model = ChatOpenAI()
humanMessage = HumanMessage("What is the capital of India")

print(model.invoke([humanMessage]))

systemMessage = SystemMessage('''You are a helpful assistant that responds with three exclamation marks.''')

print(model.invoke([systemMessage, humanMessage]))