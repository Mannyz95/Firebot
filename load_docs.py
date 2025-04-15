from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_fdny_pdfs(directory):
    all_chunks = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                loader = PyMuPDFLoader(pdf_path)
                pdf_docs = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = text_splitter.split_documents(pdf_docs)
                all_chunks.extend(chunks)

    return all_chunks
