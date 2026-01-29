import os
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

PDF_PATH = "data"          # agent-1/data/
VECTOR_DB_PATH = "index"

def chunk_text(text, chunk_size=400, overlap=80):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + chunk_size])
        i += chunk_size - overlap
    return chunks

def ingest():
    documents = []

    for file in os.listdir(PDF_PATH):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(PDF_PATH, file))
            full_text = ""

            for page in reader.pages:
                full_text += page.extract_text() or ""

            chunks = chunk_text(full_text)

            for chunk in chunks:
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={"source": file}
                    )
                )

    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )

    db = FAISS.from_documents(documents, embeddings)
    db.save_local(VECTOR_DB_PATH)

    print("✅ PDF ingestion completed successfully.")

if __name__ == "__main__":
    ingest()
