import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

def load_fdny_pdfs(folder_path):
    docs = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.pdf'):
                file_path = os.path.join(root, file)
                loader = PyPDFLoader(file_path)
                pdf_docs = loader.load()
                for doc in pdf_docs:
                    doc.metadata["source"] = file  # Track filename
                docs.extend(pdf_docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    return chunks
