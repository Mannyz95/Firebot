import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
from load_docs import load_fdny_pdfs

load_dotenv()

st.set_page_config(page_title="Firebot", page_icon="🔥")
st.title("🔥 Firebot: FDNY Study Assistant")

query = st.text_input("Ask a question about FDNY protocols, exams, or SOPs:")

if query:
    # Load FDNY docs into memory only
    chunks = load_fdny_pdfs("fire_docs/")

    # Build vectorstore in memory (no persist)
    db = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    )

    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY")),
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

