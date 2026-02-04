from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
import os

# ⚠️ NEVER hardcode API keys in production
os.environ["OPENAI_API_KEY"] = "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"

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="gpt-3.5-turbo",
    streaming=True
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
