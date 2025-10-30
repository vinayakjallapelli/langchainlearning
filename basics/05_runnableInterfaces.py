import os

from langchain_openai.chat_models import ChatOpenAI

os.getenv("OPENAI_API_KEY")
model = ChatOpenAI()

completions = model.invoke("Hi there!")
print(completions)
print('------')
completions = model.batch(['Hi there!', 'Bye!'])
print(completions)
print('------')
for token in model.stream('Bye!'):
    print(token)




