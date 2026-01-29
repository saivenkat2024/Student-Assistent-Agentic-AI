import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

VECTOR_DB_PATH = "index"

# --------------------------------------------------
# LLM (OpenRouter)
# --------------------------------------------------
llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="gpt-3.5-turbo",
    temperature=0.4
)

# --------------------------------------------------
# Embeddings
# --------------------------------------------------
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# --------------------------------------------------
# Load FAISS ONCE
# --------------------------------------------------
try:
    db = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
except Exception as e:
    raise RuntimeError(f"❌ Failed to load FAISS index: {e}")

# --------------------------------------------------
# Ask PDF Function
# --------------------------------------------------
def ask_pdf(question: str) -> str:

    """
    Answers questions using ONLY the indexed PDF content,
    with detailed, student-friendly explanations.
    """

    docs = db.max_marginal_relevance_search(
        question,
        k=8,          # ⬅️ increased
        fetch_k=30    # ⬅️ more context
    )

    if not docs:
        return "❌ Not found in the document."

    context = "\n\n".join(d.page_content for d in docs)

    if not context.strip():
        return "❌ Not found in the document."

    prompt = f"""
You are a knowledgeable academic tutor.

Your task is to answer the question using ONLY the information
present in the context below.

IMPORTANT INSTRUCTIONS:
- Be detailed and descriptive
- Explain concepts step by step
- Use simple, student-friendly language
- Clearly explain causes, reasons, and implications
- Rephrase and expand ideas found in the context
- Do NOT introduce any external knowledge
- Do NOT mention things not supported by the context

If the context does not contain enough information,
say exactly:
"Not found in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)
    return response.content.strip()

