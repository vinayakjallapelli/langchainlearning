import os

from langgraph.graph import StateGraph, START, END
from langchain_openai.chat_models import ChatOpenAI
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage

os.getenv("OPENAI_API_KEY")


class State(TypedDict):
    messages: Annotated[list, add_messages]


model = ChatOpenAI()


def chatbot(state: State):
    answer = model.invoke(state["messages"])
    print('LLM Answer: ', answer)
    return {"messages": [answer]}


builder = StateGraph(State)
builder.add_node('chatbot', chatbot)
builder.add_edge(START, 'chatbot')
builder.add_edge('chatbot', END)
graph = builder.compile(checkpointer=InMemorySaver())


thread_1 = {"configurable": {"thread_id": "1"}}

result_1 = graph.invoke({"messages": [HumanMessage("Hi! My name is Vinayak!")]}, thread_1)
print('State: ', graph.get_state(thread_1))

result_2 = graph.invoke({"messages": [HumanMessage("I like Chicken Tandoori a lot!")]}, thread_1)
print('State: ', graph.get_state(thread_1))

result_3 = graph.invoke({"messages": [HumanMessage("What's my name and what do I like?")]}, thread_1)
print('State: ', graph.get_state(thread_1))