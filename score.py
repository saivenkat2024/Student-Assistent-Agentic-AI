from typing import TypedDict
import os
import cv2
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# ==================================================
# CONFIG
# ==================================================
load_dotenv()

OCR_API_KEY = "K86032675288957"   # your OCR.space key
OCR_URL = "https://api.ocr.space/parse/image"

os.environ["OPENROUTER_API_KEY"] = "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"

llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="gpt-3.5-turbo"
)

# ==================================================
# IMAGE PREPROCESSING
# ==================================================
def preprocess_image(path: str) -> str:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    processed_path = "processed_temp.png"
    cv2.imwrite(processed_path, img)
    return processed_path

# ==================================================
# OCR USING OCR.SPACE API
# ==================================================
def ocr_space_extract(image_path: str) -> str:
    with open(image_path, "rb") as f:
        response = requests.post(
            OCR_URL,
            files={"file": f},
            data={
                "apikey": OCR_API_KEY,
                "language": "eng",
                "isOverlayRequired": False,
                "OCREngine": 2
            }
        )

    result = response.json()

    if "ParsedResults" not in result:
        raise RuntimeError("OCR failed")

    return result["ParsedResults"][0]["ParsedText"].strip()

# ==================================================
# STATE
# ==================================================
class EvalState(TypedDict):
    image_path: str
    extracted_text: str
    evaluation: str

# ==================================================
# NODES
# ==================================================
def ocr_node(state: EvalState):
    processed_img = preprocess_image(state["image_path"])
    text = ocr_space_extract(processed_img)
    return {"extracted_text": text}

def cleanup_node(state: EvalState):
    prompt = f"""
You are correcting OCR output from a handwritten exam answer.

Rules:
- Fix spelling and grammar
- Preserve technical terms (A*, g(n), h(n), f(n))
- Do NOT add new information
- Do NOT remove content

OCR Text:
{state['extracted_text']}

Cleaned Answer:
"""
    response = llm.invoke(prompt)
    return {"extracted_text": response.content.strip()}

def evaluation_node(state: EvalState):
    prompt = f"""
You are an exam evaluator.

Question:
What is the A* search algorithm?

Student Answer:
{state['extracted_text']}

Return:
- Score out of 5
- Short feedback (2–3 lines)
"""
    response = llm.invoke(prompt)
    return {"evaluation": response.content.strip()}

# ==================================================
# GRAPH
# ==================================================
graph = StateGraph(EvalState)

graph.add_node("ocr", ocr_node)
graph.add_node("cleanup", cleanup_node)
graph.add_node("evaluate", evaluation_node)

graph.set_entry_point("ocr")
graph.add_edge("ocr", "cleanup")
graph.add_edge("cleanup", "evaluate")
graph.add_edge("evaluate", END)

app = graph.compile()

# ==================================================
# RUN
# ==================================================
if __name__ == "__main__":
    result = app.invoke({
        "image_path": "hi.jpg"   # your answer image
    })

    print("\n--- CLEANED ANSWER ---")
    print(result["extracted_text"])

    print("\n--- EVALUATION ---")
    print(result["evaluation"])