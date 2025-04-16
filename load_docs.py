import gdown
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_fdny_pdfs():
    all_chunks = []
    os.makedirs("temp_docs", exist_ok=True)

    # 🔥 Google Drive file ID map
    pdfs = {
        "engine_ops.pdf": "1WV2sdHoIU_AI0IdGkQ_KI_ePw81hwfpI",
        "ladder_tactics.pdf": "1L8uJZCNq80duzQUNuk5EHh7yu89SJ_Xj",
        "cfr_manual.pdf": "1z58ih3DTLFD6C6hluNx3_Md_GoIfyQt4",
    }

    for filename, file_id in pdfs.items():
        local_path = f"temp_docs/{filename}"
        url = f"https://drive.google.com/uc?id={file_id}"
        try:
            gdown.download(url, local_path, quiet=False)

            loader = PyMuPDFLoader(local_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)

        except Exception as e:
            print(f"⚠️ Failed to load {filename}: {e}")

    return all_chunks

