import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# Suppress transformers deprecation noise when optional deps are probed.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import streamlit as st
import PyPDF2
import docx
import pandas as pd
import json
import time
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

def _env(key: str, default: str | None = None) -> str:
    return os.getenv(key, default or "")


def _env_float(key: str, default: float) -> float:
    return float(_env(key, str(default)))


def _env_int(key: str, default: int) -> int:
    return int(_env(key, str(default)))


def _hf_api_token() -> str:
    token = _env("HUGGINGFACEHUB_API_TOKEN")
    if not token or token == "your_huggingface_token_here":
        st.error(
            "Missing Hugging Face API token. Copy `.env.example` to `.env` and set "
            "`HUGGINGFACEHUB_API_TOKEN`."
        )
        st.stop()
    return token


def create_llm() -> ChatHuggingFace:
    endpoint = HuggingFaceEndpoint(
        repo_id=_env("HF_MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct"),
        huggingfacehub_api_token=_hf_api_token(),
        temperature=_env_float("HF_TEMPERATURE", 0.2),
        max_new_tokens=_env_int("HF_MAX_NEW_TOKENS", 512),
    )
    return ChatHuggingFace(llm=endpoint)

# Function to read different file types
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = "".join(page.extract_text() for page in pdf_reader.pages)
    return text

def read_word(file):
    doc = docx.Document(file)
    return "\n".join(para.text for para in doc.paragraphs)

def read_excel(file):
    df = pd.read_excel(file)
    return df.to_string()

def read_csv(file):
    df = pd.read_csv(file)
    return df.to_string()

def read_json(file):
    data = json.load(file)
    return json.dumps(data, indent=4)

def read_txt(file):
    return file.read().decode("utf-8")

# Load and process files
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
        return None
    
    text_splitter = CharacterTextSplitter(
        chunk_size=_env_int("CHUNK_SIZE", 1000),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 200),
    )
    return text_splitter.split_text(text)

# Create vector store
def create_vector_store(texts):
    embeddings = HuggingFaceEmbeddings(
        model_name=_env("EMBEDDING_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")
    )
    return FAISS.from_texts(texts, embeddings)

# Set up QA system
def setup_qa_system(vector_store):
    return RetrievalQA.from_chain_type(
        llm=create_llm(),
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
    )


def setup_general_chatbot():
    return create_llm()

# Streamlit UI
st.set_page_config(page_title="MultiDoc-ChatBot", page_icon="🤖")
st.title("MultiDoc-ChatBot 🤖")

# Sidebar for file upload
st.sidebar.title("Upload a File")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["txt", "pdf", "docx", "json", "xlsx", "csv"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1].lower()
    texts = load_and_process_file(uploaded_file, file_type)
    
    if texts:
        with st.sidebar:
            with st.spinner("Processing document..."):
                vector_store = create_vector_store(texts)
                qa_system = setup_qa_system(vector_store)
                st.session_state.qa_system = qa_system
                st.sidebar.success(f"File '{uploaded_file.name}' processed successfully!")
                
                # Show file preview
                st.sidebar.subheader("File Preview")
                st.sidebar.text_area("Contents", "\n".join(texts[:10]), height=200)
else:
    if "general_chatbot" not in st.session_state:
        st.session_state.general_chatbot = setup_general_chatbot()



# Default chat message
with st.chat_message("assistant"):
    st.write("Hello there! How can I assist you today?")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    if "qa_system" in st.session_state:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                bot_response = st.session_state.qa_system.invoke(user_input)
            except Exception as e:
                st.error(f"Could not get an answer: {e}")
                st.stop()
            
            if isinstance(bot_response, dict):
                extracted_response = bot_response.get("result", "No response available").strip()
            else:
                extracted_response = str(bot_response).strip()
            
            for i in range(len(extracted_response)):
                message_placeholder.markdown(extracted_response[:i+1])
                time.sleep(_env_float("TYPING_DELAY_DOC", 0.003))
            
            st.session_state.messages.append({"role": "assistant", "content": extracted_response})
    else:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                bot_response = st.session_state.general_chatbot.invoke(user_input)
            except Exception as e:
                st.error(f"Could not get a response: {e}")
                st.stop()
            
            if hasattr(bot_response, "content"):
                extracted_response = bot_response.content.strip()
            elif isinstance(bot_response, dict):
                extracted_response = bot_response.get("result", "No response available").strip()
            else:
                extracted_response = str(bot_response).strip()
            
            for i in range(len(extracted_response)):
                message_placeholder.markdown(extracted_response[:i+1])
                time.sleep(_env_float("TYPING_DELAY_CHAT", 0.02))
            
            st.session_state.messages.append({"role": "assistant", "content": extracted_response})

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()