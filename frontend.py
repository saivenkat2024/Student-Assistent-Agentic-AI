import streamlit as st
from backend import chatbot
from rag import ask_pdf
from langchain_core.messages import HumanMessage
import time

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="Chatbot", layout="centered")
st.title("💬 Chatbot")

# --------------------------------------------------
# Cache RAG function (CRITICAL)
# --------------------------------------------------
@st.cache_resource
def get_ask_pdf():
    return ask_pdf

pdf_ask = get_ask_pdf()

# --------------------------------------------------
# Helper: stream text safely (for PDF answers)
# --------------------------------------------------
def stream_text(text, delay=0.01):
    """
    Simulates streaming for a completed text response.
    """
    for word in text.split():
        yield word + " "
        time.sleep(delay)

# --------------------------------------------------
# Toggle for RAG Mode
# --------------------------------------------------
rag_mode = st.toggle("📄 Ask PDF (RAG Mode)", value=False)

if rag_mode:
    st.success("RAG Mode Enabled (PDF-based answers)")
else:
    st.info("Normal Chat Mode (LLM conversation)")

# --------------------------------------------------
# LangChain thread config
# --------------------------------------------------
CONFIG = {'configurable': {'thread_id': 'thread-1'}}

# --------------------------------------------------
# Session state for chat history
# --------------------------------------------------
if "message_history" not in st.session_state:
    st.session_state.message_history = []

# --------------------------------------------------
# Render previous messages
# --------------------------------------------------
for msg in st.session_state.message_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------------
# Chat input
# --------------------------------------------------
user_input = st.chat_input("Type your message...")

if user_input:
    # ---- User message ----
    st.session_state.message_history.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # ---- Assistant message ----
    with st.chat_message("assistant"):

        # ✅ PDF MODE (STREAMED)
        if rag_mode:
            with st.spinner("Searching PDF..."):
                full_answer = pdf_ask(user_input)

            # Stream the PDF answer
            ai_message = st.write_stream(
                stream_text(full_answer)
            )

        # ❌ NORMAL CHAT MODE (already streamed)
        else:
            ai_message = st.write_stream(
                chunk.content
                for chunk, _ in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="messages",
                )
            )

    # ---- Save assistant message ----
    st.session_state.message_history.append(
        {"role": "assistant", "content": ai_message}
    )
