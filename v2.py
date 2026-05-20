import json
import time

import docx
import pandas as pd
import PyPDF2
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


# =====================================================
# FILE READERS
# =====================================================


def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""

    for page in pdf_reader.pages:
        extracted = page.extract_text()

        if extracted:
            text += extracted

    return text


def read_word(file):
    doc = docx.Document(file)
    return "\n".join(para.text for para in doc.paragraphs)


def read_excel(file):
    df = pd.read_excel(file)
    return df.to_string(index=False)


def read_csv(file):
    df = pd.read_csv(file)
    return df.to_string(index=False)


def read_json(file):
    data = json.load(file)
    return json.dumps(data, indent=4)


def read_txt(file):
    return file.read().decode("utf-8")


# =====================================================
# LOAD & PROCESS DOCUMENTS
# =====================================================


def load_and_process_file(uploaded_file, file_type):
    if file_type == "pdf":
        text = read_pdf(uploaded_file)

    elif file_type == "docx":
        text = read_word(uploaded_file)

    elif file_type == "xlsx":
        text = read_excel(uploaded_file)

    elif file_type == "csv":
        text = read_csv(uploaded_file)

    elif file_type == "json":
        text = read_json(uploaded_file)

    elif file_type == "txt":
        text = read_txt(uploaded_file)

    else:
        st.error("Unsupported file type")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    return splitter.split_text(text)


# =====================================================
# VECTOR STORE
# =====================================================


def create_vector_store(texts):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    vector_store = FAISS.from_texts(texts, embeddings)

    return vector_store


# =====================================================
# QA CHAIN
# =====================================================


def setup_qa_system(vector_store):

    huggingface_api_token = "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"

    # Switched to Llama 3.2 3B Instruct on native HF inference
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.2-3B-Instruct",
        huggingfacehub_api_token=huggingface_api_token,
        temperature=0.2,
        max_new_tokens=512,
        provider="hf-inference",
    )

    prompt = ChatPromptTemplate.from_template(
        """
Answer the user's question using ONLY the provided context.

Context:
{context}

Question:
{input}

Answer:
"""
    )

    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain,
    )

    return retrieval_chain


# =====================================================
# GENERAL CHATBOT
# =====================================================


def setup_general_chatbot():

    huggingface_api_token = "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"

    # Switched to Llama 3.2 3B Instruct on native HF inference
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.2-3B-Instruct",
        huggingfacehub_api_token=huggingface_api_token,
        temperature=0.3,
        max_new_tokens=512,
        provider="hf-inference",
    )

    return llm


# =====================================================
# STREAMLIT UI
# =====================================================


st.set_page_config(
    page_title="MultiDoc-ChatBot",
    page_icon="🤖",
    layout="wide",
)

st.title("MultiDoc-ChatBot 🤖")


# =====================================================
# SIDEBAR
# =====================================================


st.sidebar.title("Upload Documents")

uploaded_file = st.sidebar.file_uploader(
    "Choose a file",
    type=["txt", "pdf", "docx", "json", "xlsx", "csv"],
)


# =====================================================
# DOCUMENT PROCESSING
# =====================================================


if uploaded_file is not None:

    file_type = uploaded_file.name.split(".")[-1].lower()

    with st.sidebar:

        with st.spinner("Processing document..."):

            texts = load_and_process_file(uploaded_file, file_type)

            vector_store = create_vector_store(texts)

            qa_system = setup_qa_system(vector_store)

            st.session_state.qa_system = qa_system

            st.success(f"{uploaded_file.name} processed successfully")

            st.subheader("Preview")

            st.text_area(
                "Content Preview",
                "\n".join(texts[:5]),
                height=250,
            )

else:

    if "general_chatbot" not in st.session_state:
        st.session_state.general_chatbot = setup_general_chatbot()


# =====================================================
# CHAT MEMORY
# =====================================================


if "messages" not in st.session_state:
    st.session_state.messages = []


# =====================================================
# DISPLAY CHAT HISTORY
# =====================================================


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =====================================================
# CHAT INPUT
# =====================================================


user_input = st.chat_input("Ask something...")


if user_input:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)


    # =========================================
    # DOCUMENT QA
    # =========================================

    if "qa_system" in st.session_state:

        with st.chat_message("assistant"):

            placeholder = st.empty()

            response = st.session_state.qa_system.invoke(
                {
                    "input": user_input
                }
            )

            answer = response.get("answer", "No answer found")

            animated_text = ""

            for char in answer:
                animated_text += char
                placeholder.markdown(animated_text)
                time.sleep(0.002)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )


    # =========================================
    # GENERAL CHATBOT
    # =========================================

    else:

        with st.chat_message("assistant"):

            placeholder = st.empty()

            response = st.session_state.general_chatbot.invoke(user_input)

            animated_text = ""

            for char in response:
                animated_text += char
                placeholder.markdown(animated_text)
                time.sleep(0.002)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )


# =====================================================
# CLEAR CHAT
# =====================================================


if st.button("Clear Chat"):

    st.session_state.messages = []

    if "qa_system" in st.session_state:
        del st.session_state.qa_system

    st.rerun()