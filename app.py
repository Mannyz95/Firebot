import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
from load_docs import load_fdny_pdfs, DOC_CATEGORIES
import random
import re 

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Firebot", page_icon="🔥")
st.title("🔥 Firebot: FDNY Study Assistant")

# Initialize session state variables
if 'quiz_questions' not in st.session_state:
    st.session_state.quiz_questions = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'selected_option' not in st.session_state:
    st.session_state.selected_option = None
if 'show_answer' not in st.session_state:
    st.session_state.show_answer = False

mode = st.radio("Choose your study mode:", ["Ask a question", "Give me a quiz"])

if mode == "Ask a question":
    query = st.text_input("Ask a question about FDNY protocols, exams, or SOPs:")

    if query:
        try:
            st.write("📄 Downloading and loading FDNY PDFs...")
            chunks = load_fdny_pdfs()

            st.write(f"📦 Loaded {len(chunks)} chunks from PDF files.")

            if len(chunks) == 0:
                st.error("⚠️ No content was extracted from the PDFs.")
                st.stop()

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

        except Exception as e:
            st.error(f"🚨 Failed to load or process PDFs: {e}")
            st.stop()

elif mode == "Give me a quiz":
    st.subheader("🧠 Take a 10-question FDNY Quiz")

    selected_topics = st.multiselect("📚 Select one or more topics:", list(DOC_CATEGORIES.keys()))

    if not selected_topics:
        st.warning("Please select at least one topic to generate quiz questions.")
        st.stop()

    chunks = load_fdny_pdfs(categories=selected_topics)
    st.write(f"📦 Loaded {len(chunks)} chunks from selected categories.")

    if not chunks:
        st.error("🚨 No documents were loaded from the selected categories.")
        st.stop()

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = FAISS.from_documents(chunks, embedding=embeddings)

    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", api_key=OPENAI_API_KEY),
        chain_type="stuff",
        retriever=retriever
    )

    quiz_prompt = """
    Generate 10 multiple-choice questions based on FDNY procedures and SOPs.
    Format like:
    1. Question?
    A) ...
    B) ...
    C) ...
    D) ...
    Correct Answer: X
    """

    quiz_text = qa.run(quiz_prompt)

    st.markdown("### 📋 Quiz")
    
    questions = re.split(r"\n(?=\d+\.)", quiz_text.strip())

    for q in questions:
        match = re.match(r"\d+\.\s*(.*?)\nA\)(.*?)\nB\)(.*?)\nC\)(.*?)\nD\)(.*?)\nCorrect Answer: ([ABCD])", q, re.DOTALL)
        if not match:
            st.warning("⚠️ Question format was not understood.")
            st.text(q)
            continue

        question_text, a, b, c, d, correct = match.groups()
        user_answer = st.radio(
            question_text.strip(),
            [f"A) {a.strip()}", f"B) {b.strip()}", f"C) {c.strip()}", f"D) {d.strip()}"],
            key=question_text
        )

        if user_answer.startswith(correct):
            st.success("✅ You got it right!")
        else:
            st.error(f"❌ Nope. The correct answer was: {correct})")

        st.markdown("---")














