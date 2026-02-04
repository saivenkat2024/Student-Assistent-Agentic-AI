import streamlit as st
from backend import chatbot
from rag import ask_pdf
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import uuid
import time
from datetime import datetime

# --------------------------------------------------
# Utilities
# --------------------------------------------------

def generate_thread_id():
    return str(uuid.uuid4())


def now_ts():
    return datetime.utcnow().timestamp()


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    ).values
    return state.get("messages", [])


# --------------------------------------------------
# Title LLM (separate from LangGraph)
# --------------------------------------------------

title_llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="gpt-3.5-turbo"
)


def generate_chat_title(user_msg, assistant_msg):
    prompt = f"""
Generate a short, clear title (3–6 words) for this conversation.
Do not use quotes.

User: {user_msg}
Assistant: {assistant_msg}
"""
    return title_llm.invoke(prompt).content.strip()


# --------------------------------------------------
# Page config
# --------------------------------------------------

st.set_page_config(page_title="Agent Chat", layout="centered")
st.title("💬 Agent")

# --------------------------------------------------
# Cache RAG
# --------------------------------------------------

@st.cache_resource
def get_ask_pdf():
    return ask_pdf

pdf_ask = get_ask_pdf()

# --------------------------------------------------
# Session state
# --------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if "chat_titles" not in st.session_state:
    st.session_state.chat_titles = {}

if "chat_last_updated" not in st.session_state:
    st.session_state.chat_last_updated = {}

if "rag_mode" not in st.session_state:
    st.session_state.rag_mode = {}

if st.session_state.thread_id not in st.session_state.rag_mode:
    st.session_state.rag_mode[st.session_state.thread_id] = False

# --------------------------------------------------
# LangGraph config (MANDATORY)
# --------------------------------------------------

CONFIG = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}

# --------------------------------------------------
# RAG toggle
# --------------------------------------------------

rag_mode = st.toggle(
    "📄 Ask PDF (RAG Mode)",
    value=st.session_state.rag_mode[st.session_state.thread_id]
)
st.session_state.rag_mode[st.session_state.thread_id] = rag_mode

# --------------------------------------------------
# Sidebar (ChatGPT style)
# --------------------------------------------------

st.sidebar.title("LangGraph Agent")

if st.sidebar.button("➕ New Chat"):
    st.session_state.thread_id = generate_thread_id()
    st.session_state.rag_mode[st.session_state.thread_id] = False
    st.rerun()

st.sidebar.header("My Conversations")

sorted_threads = sorted(
    st.session_state.chat_titles.keys(),
    key=lambda t: st.session_state.chat_last_updated.get(t, 0),
    reverse=True
)

for tid in sorted_threads:
    if st.sidebar.button(st.session_state.chat_titles[tid], key=tid):
        st.session_state.thread_id = tid
        st.rerun()

# --------------------------------------------------
# Render messages (ONLY from LangGraph)
# --------------------------------------------------

messages = load_conversation(st.session_state.thread_id)

for msg in messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# --------------------------------------------------
# Chat input
# --------------------------------------------------

user_input = st.chat_input("Type your message...")

if user_input:
    # 1️⃣ Show user instantly
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2️⃣ Save user once
    chatbot.update_state(
        CONFIG,
        {"messages": [HumanMessage(content=user_input)]}
    )

    # 3️⃣ Assistant response (CRITICAL FIX HERE)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        if rag_mode:
            with st.spinner("Searching PDF..."):
                answer = pdf_ask(user_input)

            for ch in answer:
                full_response += ch
                placeholder.markdown(full_response)
                time.sleep(0.002)

            chatbot.update_state(
                CONFIG,
                {"messages": [AIMessage(content=full_response)]}
            )
        else:
            # ✅ MUST PASS EMPTY INPUT TO TRIGGER GRAPH
            for chunk, _ in chatbot.stream(
                {},
                config=CONFIG,
                stream_mode="messages",
            ):
                full_response += chunk.content
                placeholder.markdown(full_response)

    # 4️⃣ Chat title (first message only)
    tid = st.session_state.thread_id
    if tid not in st.session_state.chat_titles:
        title = generate_chat_title(user_input, full_response)
        st.session_state.chat_titles[tid] = title

    st.session_state.chat_last_updated[tid] = now_ts()
    st.rerun()
