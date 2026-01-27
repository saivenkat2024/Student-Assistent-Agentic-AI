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
