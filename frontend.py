import streamlit as st
from backend import chatbot, generate_chat_title
from rag import ask_pdf
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import textwrap
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tkinter as tk
from tkinter import filedialog
import time

# --------------------------------------------------
# PDF Utilities
# --------------------------------------------------

def get_file_path():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.asksaveasfilename(
        defaultextension=".pdf",
        filetypes=[("PDF files", "*.pdf")],
        title="Save responses as PDF"
    )
    root.destroy()
    return path


def save_responses_to_pdf(responses, file_path):
    """
    responses = list of tuples -> [(user_msg, assistant_msg), ...]
    """
    c = canvas.Canvas(file_path, pagesize=letter)
    width, height = letter
    y = height - 50

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "CHAT RESPONSES")
    y -= 30
    c.setFont("Helvetica", 10)

    for idx, (user_msg, assistant_msg) in enumerate(responses, start=1):
        wrapped_user = textwrap.wrap(f"Q{idx}: {user_msg}", 95)
        for line in wrapped_user:
            if y < 60:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 10)
            c.drawString(50, y, line)
            y -= 14

        y -= 10

        wrapped_agent = textwrap.wrap(f"A{idx}: {assistant_msg}", 95)
        for line in wrapped_agent:
            if y < 60:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 10)
            c.drawString(50, y, line)
            y -= 14

        y -= 25

    c.save()

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(page_title="Agent Chat", layout="centered")

# --------------------------------------------------
# Session State
# --------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_titles" not in st.session_state:
    st.session_state.chat_titles = {}

if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

# 🔥 THIS IS THE KEY FIX
if "pdf_responses" not in st.session_state:
    st.session_state.pdf_responses = []  # [(user, assistant)]

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("Conversations")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.pdf_path = None
    st.session_state.pdf_responses.clear()

for tid, title in st.session_state.chat_titles.items():
    if st.sidebar.button(f"💬 {title}", key=tid, use_container_width=True):
        st.session_state.thread_id = tid
        st.session_state.pdf_path = None
        st.session_state.pdf_responses.clear()

# --------------------------------------------------
# Chat Area
# --------------------------------------------------

st.title("💬 Agent")

state = chatbot.get_state(
    config={"configurable": {"thread_id": st.session_state.thread_id}}
).values

messages = state.get("messages", [])

for i, msg in enumerate(messages):
    role = "user" if isinstance(msg, HumanMessage) else "assistant"

    with st.chat_message(role):
        st.markdown(msg.content)

        if role == "assistant":
            cols = st.columns([0.06, 0.06, 0.06, 0.22, 0.6])

            cols[0].button("👍", key=f"up_{i}")
            cols[1].button("👎", key=f"down_{i}")
            cols[2].button("📋", key=f"copy_{i}")

            if cols[3].button("📥 Download", key=f"dl_{i}"):

                # Get matching user question
                user_text = "N/A"
                if i > 0 and isinstance(messages[i - 1], HumanMessage):
                    user_text = messages[i - 1].content

                assistant_text = msg.content

                # Avoid duplicate appends
                entry = (user_text, assistant_text)
                if entry not in st.session_state.pdf_responses:
                    st.session_state.pdf_responses.append(entry)

                if not st.session_state.pdf_path:
                    st.session_state.pdf_path = get_file_path()

                if st.session_state.pdf_path:
                    save_responses_to_pdf(
                        st.session_state.pdf_responses,
                        st.session_state.pdf_path
                    )
                    st.toast("Response appended to PDF")

# --------------------------------------------------
# Controls
# --------------------------------------------------

rag_mode = st.toggle("📄 RAG Mode", value=False)
user_input = st.chat_input("Type your message...")

# --------------------------------------------------
# Message Handling (Streaming Feel)
# --------------------------------------------------

# --------------------------------------------------
# Message Handling (IMPROVED UX)
# --------------------------------------------------

if user_input:
    # Save title once
    if st.session_state.thread_id not in st.session_state.chat_titles:
        st.session_state.chat_titles[
            st.session_state.thread_id
        ] = generate_chat_title(user_input)

    # 1️⃣ Immediately show user message (prevents disappearance)
    with st.chat_message("user"):
        st.markdown(user_input)

    # Add user message to state
    chatbot.update_state(
        {"configurable": {"thread_id": st.session_state.thread_id}},
        {"messages": [HumanMessage(content=user_input)]}
    )

    # 2️⃣ Show temporary "Thinking..." assistant bubble
    with st.chat_message("assistant"):
        thinking_box = st.empty()
        thinking_box.markdown("⏳ **Thinking…**")

        streamed_text = ""

        # Generate response
        if rag_mode:
            full_response = ask_pdf(user_input)
            chatbot.update_state(
                {"configurable": {"thread_id": st.session_state.thread_id}},
                {"messages": [AIMessage(content=full_response)]}
            )
        else:
            chatbot.invoke(
                {"messages": []},
                config={"configurable": {"thread_id": st.session_state.thread_id}}
            )

            state = chatbot.get_state(
                config={"configurable": {"thread_id": st.session_state.thread_id}}
            ).values

            full_response = state["messages"][-1].content

        # 3️⃣ Replace thinking box with streaming response
        for word in full_response.split():
            streamed_text += word + " "
            thinking_box.markdown(streamed_text)
            time.sleep(0.03)

    st.rerun()

