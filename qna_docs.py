import torch
import streamlit as st
from transformers import pipeline

# Initialize the question-answering pipeline
try:
    question_answer = pipeline("question-answering", model="deepset/roberta-base-squad2")
except Exception as e:
    st.error(f"Error initializing pipeline: {e}")


def read_file_content(file):
    """
    Reads the content of a file object and returns it.
    Parameters:
    file (file object): The uploaded file.
    Returns:
    str: The content of the file.
    """
    try:
        return file.getvalue().decode("utf-8")
    except Exception as e:
        return f"An error occurred while reading the file: {e}"


def get_answer(context, question):
    """
    Retrieves the answer to the given question based on the provided context.
    Parameters:
    context (str): The context text.
    question (str): The question to be answered.
    Returns:
    str: The answer to the question.
    """
    try:
        answer = question_answer(question=question, context=context)
        return answer["answer"]
    except Exception as e:
        return f"An error occurred while processing the question: {e}"


# Streamlit Layout
st.title("@GenAILearniverse Project 5: Document Q&A")
st.write("This application will answer questions based on the context provided in the uploaded document.")

# File uploader
uploaded_file = st.file_uploader("Upload your file", type=["txt"], help="Upload a .txt file containing the context.")
question = st.text_input("Input your question", help="Enter the question you want answered based on the file content.")

# Answer button and display
if st.button("Get Answer"):
    if uploaded_file and question.strip():
        context = read_file_content(uploaded_file)
        if context.startswith("An error occurred"):
            st.error(context)
        else:
            with st.spinner("Processing your question..."):
                answer = get_answer(context, question)
                st.text_area("Answer", answer, height=100)
    else:
        st.warning("Please upload a file and enter a question.")
