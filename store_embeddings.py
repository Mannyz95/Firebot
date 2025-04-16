from dotenv import load_dotenv
import os
load_dotenv()

print("ENV KEY FOUND:", os.getenv("OPENAI_API_KEY"))  # Debugging line

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from load_docs import load_fdny_pdfs

def store_embeddings():
    print("📄 Loading docs...")
    chunks = load_fdny_pdfs()  # ← No argument, Google Drive powered!

    # TESTING ONLY - reduce chunks to something small
    chunks = chunks[:100]  # ← Only embed the first 100 to avoid OpenAI rage

    print(f"✅ Loaded {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    vectordb.persist()
    print("🔥 FDNY memory stored.")

if __name__ == "__main__":
    store_embeddings()


