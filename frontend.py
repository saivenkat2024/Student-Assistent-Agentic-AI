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

# 👉 Import Exam Mode
from exam_mode import run_exam_mode


# --------------------------------------------------
# Session State Init
# --------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_titles" not in st.session_state:
    st.session_state.chat_titles = {}

if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

if "pdf_responses" not in st.session_state:
    st.session_state.pdf_responses = []

if "exam_mode" not in st.session_state:
    st.session_state.exam_mode = False


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
    c = canvas.Canvas(file_path, pagesize=letter)
    width, height = letter
    y = height - 50

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "CHAT RESPONSES")
    y -= 30
    c.setFont("Helvetica", 10)

    for idx, (user_msg, assistant_msg) in enumerate(responses, start=1):
        for line in textwrap.wrap(f"Q{idx}: {user_msg}", 95):
            if y < 60:
                c.showPage()
                y = height - 50
            c.drawString(50, y, line)
            y -= 14

        y -= 10

        for line in textwrap.wrap(f"A{idx}: {assistant_msg}", 95):
            if y < 60:
                c.showPage()
                y = height - 50
            c.drawString(50, y, line)
            y -= 14

        y -= 25

    c.save()


# --------------------------------------------------
# PAGE SWITCH: EXAM MODE
# --------------------------------------------------

if st.session_state.exam_mode:
    run_exam_mode()

    if st.button("⬅ Back to Chat"):
        st.session_state.exam_mode = False
        st.rerun()

    st.stop()


# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(page_title="Agent Chat", layout="wide")


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
# Layout
# --------------------------------------------------

left_col, right_col = st.columns([0.75, 0.25])


# --------------------------------------------------
# RIGHT PANEL — TOOLS
# --------------------------------------------------

with right_col:
    st.markdown("## 🎓 Tools")

    if st.button("📝 Exam Mode", use_container_width=True):
        st.session_state.exam_mode = True
        st.rerun()


# --------------------------------------------------
# CHAT DISPLAY (NO INPUT HERE)
# --------------------------------------------------

with left_col:
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
                    user_text = messages[i - 1].content if i > 0 else "N/A"
                    assistant_text = msg.content
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

    rag_mode = st.toggle("📄 RAG Mode", value=False)


# --------------------------------------------------
# CHAT INPUT — MUST BE LAST
# --------------------------------------------------

user_input = st.chat_input("Type your message...")


# --------------------------------------------------
# INPUT HANDLER
# --------------------------------------------------

if user_input:
    if st.session_state.thread_id not in st.session_state.chat_titles:
        st.session_state.chat_titles[
            st.session_state.thread_id
        ] = generate_chat_title(user_input)

    with st.chat_message("user"):
        st.markdown(user_input)

    chatbot.update_state(
        {"configurable": {"thread_id": st.session_state.thread_id}},
        {"messages": [HumanMessage(content=user_input)]}
    )

    with st.chat_message("assistant"):
        thinking_box = st.empty()
        thinking_box.markdown("⏳ **Thinking…**")

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

        streamed_text = ""
        for word in full_response.split():
            streamed_text += word + " "
            thinking_box.markdown(streamed_text)
            time.sleep(0.02)

    st.rerun()
