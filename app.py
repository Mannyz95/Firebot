import streamlit as st
from langchain_community.vectorstores import FAISS
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

mode = st.radio("Choose your study mode:", ["Ask a question", "Give me a quiz"])

try:
    if mode == "Ask a question":
        query = st.text_input("Ask a question about FDNY protocols, exams, or SOPs:")

        if query:
            st.write("📄 Downloading and loading FDNY PDFs...")
            chunks = load_fdny_pdfs()
            st.write(f"📦 Loaded {len(chunks)} chunks from PDF files.")

            if len(chunks) == 0:
                st.error("⚠️ No content was extracted from the PDFs.")
                st.stop()

            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

            # Optional ladder-related filtering
            if any(word in query.lower() for word in ["ladder", "truck", "ventilation"]):
                st.write("🔍 Ladder-related query detected — filtering for ladder/tactics content...")
                chunks = [doc for doc in chunks if "ladder" in doc.metadata.get("source", "").lower()]
                if not chunks:
                    st.warning("⚠️ No ladder-specific docs found. Using full set again.")
                    chunks = load_fdny_pdfs()

            db = FAISS.from_documents(documents=chunks, embedding=embeddings)
            retriever = db.as_retriever()

            qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model_name="gpt-3.5-turbo", api_key=OPENAI_API_KEY),
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )

            result = qa(query)
            st.markdown("### 💬 Answer:")
            st.write(result["result"])

            sources = set()
            for doc in result["source_documents"]:
                filename = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "??")
                sources.add(f"{filename}, page {page}")

            if sources:
                st.markdown("### 📚 Sources:")
                for src in sources:
                    st.markdown(f"- {src}")

    elif mode == "Give me a quiz":
        st.write("🧠 Generating a multiple choice quiz question...")

        chunks = load_fdny_pdfs()
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        db = FAISS.from_documents(chunks, embedding=embeddings)

        quiz_prompt = (
            "Create one multiple choice question based on FDNY firefighting protocols, "
            "ladder company operations, or engine company duties. Include four answer choices "
            "labeled A through D, and indicate the correct answer at the end."
        )

        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", api_key=OPENAI_API_KEY),
            chain_type="stuff",
            retriever=db.as_retriever()
        )

        quiz = qa.run(quiz_prompt)
        st.markdown("### 📋 Quiz Question")
        st.markdown(quiz)

except Exception as e:
    st.error(f"🚨 Firebot blew up: {str(e)}")
    st.stop()








