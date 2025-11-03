import os
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai.chat_models import ChatOpenAI

os.getenv("OPENAI_API_KEY")


class State(TypedDict):
    messages: Annotated[list, add_messages]


model = ChatOpenAI()


def chatbot(state: State):
    print('State before: ', state)
    answer = model.invoke(state["messages"])
    print('Answer: ', answer)
    print('State after: ', state)
    return {"messages": [answer]}


# build graph
builder = StateGraph(State)
builder.add_node('chatbot', chatbot)
builder.add_edge(START, 'chatbot')
builder.add_edge('chatbot', END)
graph = builder.compile()

# print mermaid graph
print(graph.get_graph().draw_mermaid())

# sample user query
input = {"messages": [HumanMessage('hi!')]}

# invoke graph and capture output
for chunk in graph.invoke().stream(input):
    print(chunk)
