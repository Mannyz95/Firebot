import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from load_docs import load_fdny_pdfs, DOC_CATEGORIES

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="🔥 Firebot", page_icon="🚒")
st.title("🔥 Firebot: FDNY Study Assistant")

mode = st.radio("Choose your study mode:", ["Ask a question", "Give me a quiz"])

selected_categories = st.multiselect("Choose document categories to include:", list(DOC_CATEGORIES.keys()))

if st.button("Start"):
    if not selected_categories:
        st.warning("Please select at least one category.")
        st.stop()

    st.write("📄 Downloading and loading FDNY PDFs...")
    chunks = load_fdny_pdfs(categories=selected_categories)
    st.write(f"📦 Loaded {len(chunks)} chunks from PDF files.")

    if len(chunks) == 0:
        st.error("⚠️ No content was extracted from the PDFs.")
        st.stop()

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = FAISS.from_documents(documents=chunks, embedding=embeddings)
    retriever = db.as_retriever()

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", api_key=OPENAI_API_KEY),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    if mode == "Ask a question":
        query = st.text_input("Ask a question about FDNY protocols, exams, or SOPs:")
        if query:
            result = qa(query)
            st.markdown("### 💬 Answer:")
            st.write(result["result"])

            sources = {
                f"{doc.metadata.get('source', 'unknown')}, page {doc.metadata.get('page', '??')}"
                for doc in result["source_documents"]
            }

            if sources:
                st.markdown("### 📚 Sources:")
                for src in sources:
                    st.markdown(f"- {src}")

    elif mode == "Give me a quiz":
        st.write("🧠 Generating a multiple choice quiz question...")

        quiz_prompt = (
            "Create one multiple choice question based on FDNY firefighting protocols, "
            "ladder company operations, or engine company duties. Include four answer choices "
            "labeled A through D, and indicate the correct answer at the end."
        )
        quiz = qa.run(quiz_prompt)
        st.markdown("### 📋 Quiz Question")
        st.markdown(quiz)












