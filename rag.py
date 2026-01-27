import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

VECTOR_DB_PATH = "index"

llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="gpt-3.5-turbo",
    temperature=0.4
)

embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

db = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

def ask_pdf(question):
    docs = db.similarity_search(question, k=6)

    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
You are a helpful academic assistant.

Answer the question using the context below.
Write a clear, well-structured, descriptive explanation.
You may rephrase and explain in your own words,
but do NOT add information not supported by the context.

If the answer is not present, say "Not found in the document".

Context:
{context}

Question:
{question}
"""

    return llm.invoke(prompt).content
