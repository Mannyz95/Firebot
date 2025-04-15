import streamlit as st
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
from load_docs import load_fdny_pdfs

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Firebot", page_icon="🔥")
st.title("🔥 Firebot: FDNY Study Assistant")

query = st.text_input("Ask a question about FDNY protocols, exams, or SOPs:")

if query:
    chunks = load_fdny_pdfs("fire_docs/")
    st.write(f"✅ Loaded {len(chunks)} chunks from PDF")
if len(chunks) == 0:
    st.error("⚠️ No content was extracted from the PDFs. Check that your fire_docs folder is present and contains readable PDFs.")

    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", api_key=OPENAI_API_KEY),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    result = qa(query)
    st.write(result["result"])

    sources = set()
    for doc in result["source_documents"]:
        filename = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "??")
        sources.add(f"{filename}, page {page}")

    if sources:
        st.markdown("**Sources:**")
        for src in sources:
            st.markdown(f"- {src}")




