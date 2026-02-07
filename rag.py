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
    temperature=0.1  # 🔑 very low → prevents summarization
)

# --------------------------------------------------
# Embeddings
# --------------------------------------------------
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# --------------------------------------------------
# Load FAISS
# --------------------------------------------------
db = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

# --------------------------------------------------
# Query normalizer (case + typo tolerant via LLM)
# --------------------------------------------------
def normalize_query(query: str) -> str:
    """
    Fix spelling and casing but keep meaning.
    """
    prompt = f"""
Correct spelling and casing of the following question.
Do NOT change its meaning.

Question:
{query}

Corrected question:
"""
    return llm.invoke(prompt).content.strip()

# --------------------------------------------------
# MAIN RAG FUNCTION (FINAL + FULL ANSWERS)
# --------------------------------------------------
def ask_pdf(question: str) -> str:
    """
    FULL-ANSWER RAG for theory PDFs.
    Extracts ALL relevant content instead of summarizing.
    """

    # 1️⃣ Normalize query (handles typos like attensiii)
    clean_question = normalize_query(question)

    # 2️⃣ Retrieve aggressively
    docs = db.similarity_search(
        clean_question,
        k=12  # ⬅️ more chunks = more content
    )

    if not docs:
        return "Not found in the document."

    # 3️⃣ Build FULL context (NO filtering)
    context = "\n\n".join(d.page_content for d in docs)

    if not context.strip():
        return "Not found in the document."

    # --------------------------------------------------
    # 🔥 EXTRACTION-BASED PROMPT (THIS IS THE KEY)
    # --------------------------------------------------
    prompt = f"""
You are an academic assistant.

TASK:
Extract and present ALL information from the context
that answers the question below.

CRITICAL RULES:
- DO NOT summarize
- DO NOT shorten
- DO NOT skip details
- If multiple paragraphs explain the concept, include ALL of them
- Rephrase lightly ONLY for clarity
- Use ONLY the given context
- If the answer does not exist, say exactly:
  "Not found in the document."

Context:
{context}

Question:
{clean_question}

Answer (full explanation from document):
"""

    response = llm.invoke(prompt)
    return response.content.strip()
