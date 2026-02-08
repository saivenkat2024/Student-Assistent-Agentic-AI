
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
    temperature=0.1
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
# Query normalizer
# --------------------------------------------------
def normalize_query(query: str) -> str:
    prompt = f"""
Correct spelling and casing of the following question.
Do NOT change its meaning.

Question:
{query}

Corrected question:
"""
    return llm.invoke(prompt).content.strip()

# --------------------------------------------------
# MAIN RAG FUNCTION (FULL EXTRACTION)
# --------------------------------------------------
def ask_pdf(question: str) -> str:
    clean_question = normalize_query(question)

    docs = db.similarity_search(
        clean_question,
        k=20
    )

    if not docs:
        return "Not found in the document."

    # --------------------------------------------------
    # Sort chunks to restore document flow
    # --------------------------------------------------
    docs = sorted(
        docs,
        key=lambda d: (
            d.metadata.get("source", ""),
            d.metadata.get("page", 0),
            d.metadata.get("chunk", 0)
        )
    )

    context = "\n\n".join(d.page_content for d in docs)

    if not context.strip():
        return "Not found in the document."

    # --------------------------------------------------
    # Extraction-only prompt
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
- Include ALL paragraphs that explain the concept
- Light rephrasing only for clarity
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