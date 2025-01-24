import os
import json
import pandas as pd
import streamlit as st
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from langchain.docstore.document import Document
import torch

# Step 1: Document Loading Logic
def load_document(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".json"):
        with open(file_path, "r") as f:
            data = json.load(f)
        text = json.dumps(data, indent=2)
        return [Document(page_content=text)], None
    elif file_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)
        text = df.to_string(index=False)
        return [Document(page_content=text)], df
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        text = df.to_string(index=False)
        return [Document(page_content=text)], df
    else:
        with open(file_path, "r") as f:
            text = f.read()
        return [Document(page_content=text)], None
    return loader.load(), None

# Step 2: Text Splitting
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# Step 3: Vector Store Creation
def create_vector_store(texts):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(texts, embeddings)

# Step 4: Load QA Model
def load_qa_model():
    model_name = "deepset/roberta-base-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return pipeline("question-answering", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Step 5: Generate Detailed Answer
def get_detailed_answer(context, query, answer):
    context_lines = context.split(". ")
    relevant_context = [line for line in context_lines if answer in line]
    explanation = ". ".join(relevant_context)
    
    if explanation:
        return f"**{answer.capitalize()}**: {explanation.strip()}"
    return f"**{answer.capitalize()}**."

# Step 6: Get Answer from Vector Store
def get_answer_from_vector_store(vector_store, query, df=None):
    if df is not None:
        structured_answer = perform_structured_query(df, query)
        if structured_answer:
            return structured_answer

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.get_relevant_documents(query)
    context = " ".join([doc.page_content for doc in relevant_docs])

    qa_pipeline = load_qa_model()
    inputs = {"question": query, "context": context}
    result = qa_pipeline(question=query, context=context)

    answer = result.get("answer", "").strip()
    if not answer:
        return "I'm not sure about that. Could you clarify your question or try something else?"

    return get_detailed_answer(context, query, answer)

# Step 7: Perform Structured Query
def perform_structured_query(df, query):
    try:
        if "find" in query.lower():
            column_name = query.split("find")[-1].strip().split(" ")[0]
            if column_name in df.columns:
                return df[column_name].to_string(index=False)
        return None
    except Exception as e:
        return f"Error in structured query: {e}"

# Step 8: Search External Resources
def search_external_resources(query):
    url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json"
    response = requests.get(url)
    data = response.json()
    if data.get("query", {}).get("search"):
        return data["query"]["search"][0]["snippet"]
    return "No relevant information found in external resources."

# Step 9: Main Query Handling Logic
def main(file_path, query):
    documents, df = load_document(file_path)
    texts = split_text(documents)
    vector_store = create_vector_store(texts)

    answer = get_answer_from_vector_store(vector_store, query, df)
    if "I'm not sure" in answer:
        external_answer = search_external_resources(query)
        return f"Answer not found in the document. External resource says: {external_answer}"
    return answer

# Streamlit frontend

st.title("Enhanced Document Bot")
st.sidebar.title("File Upload")

uploaded_file = st.sidebar.file_uploader("Upload a file", type=["pdf", "docx", "json", "csv", "xlsx", "txt"])

if uploaded_file:
    file_path = os.path.join("/tmp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"File '{uploaded_file.name}' uploaded successfully!")

    # User input for query
    query = st.text_input("Ask your question:")

    if query:
        # Main bot logic
        answer = main(file_path, query)
        st.write("Answer:", answer)

    # Option to clear the chat
    if st.button("Clear Chat"):
        st.experimental_rerun()

else:
    st.warning("Please upload a document to interact with the bot.")
