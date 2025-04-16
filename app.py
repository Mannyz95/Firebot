import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
from load_docs import load_fdny_pdfs, DOC_CATEGORIES
import random

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
    st.write("🧠 Generating a multiple-choice quiz...")

    selected_topics = st.multiselect(
        "📚 Select one or more topics for your quiz:",
        options=list(DOC_CATEGORIES.keys()),
        default=list(DOC_CATEGORIES.keys())[:1]  # Select first one by default
    )

    if not selected_topics:
        st.warning("☝️ Please select at least one topic.")
        st.stop()

    if not st.session_state.quiz_questions:
        try:
            chunks = load_fdny_pdfs(categories=selected_topics)
            st.write(f"📦 Loaded {len(chunks)} chunks from selected categories.")

            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            db = FAISS.from_documents(chunks, embedding=embeddings)

            retriever = db.as_retriever()

            qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model_name="gpt-3.5-turbo", api_key=OPENAI_API_KEY),
                chain_type="stuff",
                retriever=retriever
            )

            # Generate 10 questions
            for _ in range(10):
                quiz_prompt = (
                    "Create one multiple choice quiz question based on FDNY procedures, SOPs, or fireground tactics. "
                    "Format it like:\n"
                    "Question: <question text>\n"
                    "1. Option A\n"
                    "2. Option B\n"
                    "3. Option C\n"
                    "4. Option D\n"
                    "Correct Answer: <number of correct option>"
                )
                quiz = qa.run(quiz_prompt)
                st.session_state.quiz_questions.append(quiz)

        except Exception as e:
            st.error(f"🚨 Failed to load or process PDFs:\n\n{str(e)}")
            st.stop()

    if st.session_state.current_question < len(st.session_state.quiz_questions):
        question = st.session_state.quiz_questions[st.session_state.current_question]
        st.markdown(f"### Question {st.session_state.current_question + 1}")
        st.write(question)

        options = ["1", "2", "3", "4"]
        st.session_state.selected_option = st.radio("Select your answer:", options)

        if st.button("Submit Answer"):
            # Extract the correct answer from the question text
            correct_answer_line = [line for line in question.split('\n') if "Correct Answer:" in line]
            if correct_answer_line:
                correct_answer = correct_answer_line[0].split("Correct Answer:")[1].strip()
                if st.session_state.selected_option == correct_answer:
                    st.success("✅ Correct!")
                    st.session_state.score += 1
                else:
                    st.error(f"❌ Incorrect. The correct answer was: {correct_answer}")
            else:
                st.warning("⚠️ Could not determine the correct answer.")

            st.session_state.current_question += 1
            st.session_state.selected_option = None
            st.experimental_rerun()
    else:
        st.markdown("### 🎉 Quiz Completed!")
        st.write(f"Your final score is: {st.session_state.score} out of {len(st.session_state.quiz_questions)}")

        if st.button("Restart Quiz"):
            st.session_state.quiz_questions = []
            st.session_state.current_question = 0
            st.session_state.score = 0
            st.session_state.selected_option = None
            st.experimental_rerun()














