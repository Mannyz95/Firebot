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
import uuid
import hashlib
import pickle
import streamlit as st

@st.cache_resource(show_spinner="⚙️ Caching FAISS vectorstore...")
def get_cached_vectorstore(chunks, api_key):
    # Generate a unique hash for the current set of chunks
    combined_text = ''.join([doc.page_content for doc in chunks])
    hash_key = hashlib.sha256(combined_text.encode()).hexdigest()

    # Use the hash to store/retrieve FAISS DB
    if 'faiss_cache' not in st.session_state:
        st.session_state.faiss_cache = {}

    if hash_key in st.session_state.faiss_cache:
        return st.session_state.faiss_cache[hash_key]

    # Otherwise, generate new vectorstore
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.from_documents(chunks, embedding=embeddings)
    st.session_state.faiss_cache[hash_key] = db
    return db


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

    # Only regenerate questions if empty or user resets
    if not st.session_state.quiz_questions:
        chunks = load_fdny_pdfs(categories=selected_topics)
        st.write(f"📦 Loaded {len(chunks)} chunks from selected categories.")

        if not chunks:
            st.error("🚨 No documents were loaded from the selected categories.")
            st.stop()

        db = get_cached_vectorstore(chunks, api_key=OPENAI_API_KEY)


        retriever = db.as_retriever()
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", api_key=OPENAI_API_KEY),
            chain_type="stuff",
            retriever=retriever
        )

        quiz_prompt = """
        Generate 10 multiple-choice questions based on FDNY procedures and SOPs.
        Each question should follow this format:
        1. Question?
        A) ...
        B) ...
        C) ...
        D) ...
        Correct Answer: X
        """

        quiz_text = qa.run(quiz_prompt)
        raw_questions = re.split(r"\n(?=\d+\.)", quiz_text.strip())

        for q in raw_questions:
            match = re.match(
                r"\d+\.\s*(.*?)\nA\)(.*?)\nB\)(.*?)\nC\)(.*?)\nD\)(.*?)\nCorrect Answer:\s*([ABCD])",
                q.strip(), re.DOTALL
            )
            if match:
                question_text, a, b, c, d, correct = match.groups()
                st.session_state.quiz_questions.append({
                    "question": question_text.strip(),
                    "options": {
                        "A": a.strip(),
                        "B": b.strip(),
                        "C": c.strip(),
                        "D": d.strip()
                    },
                    "correct": correct
                })

    st.markdown("### 📋 Quiz Time!")

    for i, q in enumerate(st.session_state.quiz_questions):
        st.markdown(f"**{i+1}. {q['question']}**")
        user_choice = st.radio(
            label="",
            options=[f"A) {q['options']['A']}", f"B) {q['options']['B']}", f"C) {q['options']['C']}", f"D) {q['options']['D']}"],
            key=f"q_{i}_choice"
        )

        if st.button(f"Reveal Answer {i+1}", key=f"reveal_{i}"):
            user_letter = user_choice[0]
            if user_letter == q["correct"]:
                st.success(f"✅ Correct! The answer is {q['correct']}) {q['options'][q['correct']]}")
            else:
                st.error(f"❌ Incorrect. You chose {user_letter}) but the correct answer is {q['correct']}) {q['options'][q['correct']]}")

        st.markdown("---")

    if st.button("🔁 Reset Quiz"):
        st.session_state.quiz_questions = []
        st.experimental_rerun()















