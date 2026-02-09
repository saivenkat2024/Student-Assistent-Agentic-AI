# exam_mode.py

import streamlit as st
import time
from openai import OpenAI

# ---------------- CONFIG ----------------
LLM_API_KEY = "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"
MODEL_NAME = "openai/gpt-4o-mini"

client = OpenAI(
    api_key=LLM_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

QUESTION = "What is A* search algorithm?"

# ---------------- RESET ON FIRST LOAD ----------------
if "initialized" not in st.session_state:
    st.session_state.clear()
    st.session_state.initialized = True

# ---------------- SESSION STATE ----------------
st.session_state.setdefault("exam_started", False)
st.session_state.setdefault("total_questions", 0)
st.session_state.setdefault("time_per_q", 0)
st.session_state.setdefault("current_q", 1)
st.session_state.setdefault("answers", [])
st.session_state.setdefault("start_time", None)
st.session_state.setdefault("auto_saved", False)
st.session_state.setdefault("timer_running", False)

# ---------------- LLM EVALUATION ----------------
def evaluate_all_answers(question, answers):
    compiled = ""
    for i, ans in enumerate(answers, 1):
        compiled += f"\nAnswer {i}:\n{ans}\n"

    prompt = f"""
You are an exam evaluator.

Question:
{question}

Student Answers:
{compiled}

Give:
- Score for each answer (out of 10)
- Total score
- Overall feedback (3–4 lines)
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content

# ---------------- SAVE & NEXT ----------------
def save_and_next(answer):
    # HARD GUARD — never exceed total_questions
    if len(st.session_state.answers) < st.session_state.total_questions:
        st.session_state.answers.append(answer)

    st.session_state.current_q += 1
    st.session_state.start_time = time.time()
    st.session_state.auto_saved = False

    # Stop exam exactly at limit
    if st.session_state.current_q > st.session_state.total_questions:
        st.session_state.timer_running = False
    else:
        st.rerun()

# ---------------- MAIN APP ----------------
def run_exam_mode():
    st.set_page_config(page_title="Exam Mode", layout="centered")
    st.title("📝 Exam Mode")

    # -------- CONFIG (ONLY BEFORE START) --------
    if not st.session_state.exam_started:
        tq = st.selectbox("Number of Questions", [1, 2, 3, 5])
        tpq = st.selectbox("Time per Question (minutes)", [1, 2, 5, 10])

        if st.button("🚀 Start Exam"):
            st.session_state.total_questions = tq
            st.session_state.time_per_q = tpq
            st.session_state.current_q = 1
            st.session_state.answers = []
            st.session_state.start_time = time.time()
            st.session_state.exam_started = True
            st.session_state.timer_running = True
            st.rerun()
        return

    # -------- HARD STOP (CRITICAL FIX) --------
    if st.session_state.current_q > st.session_state.total_questions:
        st.success("✅ All answers saved!")

        if st.button("📊 Final Evaluation"):
            evaluation = evaluate_all_answers(
                QUESTION,
                st.session_state.answers
            )
            st.markdown("## 📊 Evaluation")
            st.write(evaluation)
            st.success("🎉 Exam Completed!")
        return

    # -------- QUESTION UI --------
    st.subheader(
        f"Question {st.session_state.current_q} / {st.session_state.total_questions}"
    )
    st.info(QUESTION)

    elapsed = time.time() - st.session_state.start_time
    remaining = max(
        0,
        int(st.session_state.time_per_q * 60 - elapsed)
    )

    st.warning(f"⏱ Time left: {remaining} seconds")

    answer = st.text_area(
        "Type your answer",
        disabled=remaining == 0
    )

    # -------- SAVE ANYTIME --------
    if st.button("💾 Save & Next"):
        save_and_next(answer)

    # -------- AUTO SAVE ON TIMEOUT --------
    if remaining == 0 and not st.session_state.auto_saved:
        st.session_state.auto_saved = True
        save_and_next(answer)

    # -------- TIMER LOOP --------
    if st.session_state.timer_running:
        time.sleep(1)
        st.rerun()

# ---------------- RUN ----------------
if __name__ == "__main__":
    run_exam_mode()
