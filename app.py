import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="Firebot", page_icon="🔥")
st.title("🔥 Firebot: FDNY Study Assistant")

query = st.text_input("Ask a question about FDNY protocols, exams, or SOPs:")

if query:
    db = Chroma(
        persist_directory="chroma_db",
        embedding_function=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
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
