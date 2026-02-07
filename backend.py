import os
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler

# --------------------------------------------------
# ENV
# --------------------------------------------------

os.environ["OPENAI_API_KEY"] = "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"  # keep yours

# --------------------------------------------------
# LLM (DEFINED ONCE)
# --------------------------------------------------

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="gpt-3.5-turbo",
    streaming=True
)

# --------------------------------------------------
# State
# --------------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# --------------------------------------------------
# Streaming Callback
# --------------------------------------------------

class StreamHandler(BaseCallbackHandler):
    def __init__(self, on_token):
        self.on_token = on_token
        self.buffer = []

    def on_llm_new_token(self, token: str, **kwargs):
        self.buffer.append(token)
        self.on_token(token)

# --------------------------------------------------
# Graph Node
# --------------------------------------------------

def chat_node(state: ChatState):
    messages = state["messages"]

    streamed_tokens = []

    def token_callback(token):
        streamed_tokens.append(token)

    handler = StreamHandler(token_callback)

    # IMPORTANT: pass callbacks via config, NOT bind()
    llm.invoke(
        messages,
        config={"callbacks": [handler]}
    )

    full_response = "".join(streamed_tokens)

    return {"messages": [AIMessage(content=full_response)]}


# --------------------------------------------------
# Graph
# --------------------------------------------------

checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# --------------------------------------------------
# Title Generator
# --------------------------------------------------

def generate_chat_title(user_input: str) -> str:
    try:
        title_llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            model="gpt-3.5-turbo"
        )
        res = title_llm.invoke(f"Create a 3-word title for: {user_input}")
        return res.content.strip().replace('"', '')
    except Exception:
        return user_input[:20]
