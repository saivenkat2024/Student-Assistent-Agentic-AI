from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
import os

# ✅ Set key as OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = "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"

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="gpt-3.5-turbo"
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)
