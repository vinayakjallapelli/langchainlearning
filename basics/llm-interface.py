from langchain_openai.llms import OpenAI
import os
os.getenv("OPENAI_API_KEY")

model = OpenAI(model="gpt-4o-mini")
print(model.invoke("The sky is "))