import streamlit as st
import base64
import json
import pandas as pd
from docx import Document
from transformers import pipeline
import time

# Load Hugging Face's T5 model for text generation
generator = pipeline("text2text-generation", model="t5-small")

# Function to show PDF in the sidebar
def show_pdf(pdf_file, width=300, height=200):
    pdf_data = pdf_file.read()
    pdf_base64 = base64.b64encode(pdf_data).decode("utf-8")
    pdf_embed_code = f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="{width}" height="{height}" type="application/pdf"></iframe>'
    st.sidebar.markdown(pdf_embed_code, unsafe_allow_html=True)

# Function to show text files in the sidebar
def show_text(file, width=300, height=200):
    content = file.read().decode("utf-8")
    html_content = f'<textarea style="width:{width}px;height:{height}px;">{content[:1000]}</textarea>'
    st.sidebar.markdown(html_content, unsafe_allow_html=True)
    return content

# Function to show JSON files in the sidebar
def show_json(file, width=300, height=200):
    content = json.load(file)
    json_str = json.dumps(content, indent=4)
    html_content = f'<textarea style="width:{width}px;height:{height}px;">{json_str[:2000]}</textarea>'
    st.sidebar.markdown(html_content, unsafe_allow_html=True)
    return content

# Function to show data files (CSV/Excel) in the sidebar
def show_data(file, file_type, width=300, height=200):
    if file_type == "csv":
        df = pd.read_csv(file)
    elif file_type == "xlsx":
        df = pd.read_excel(file)
    st.sidebar.dataframe(df.head(), width=width, height=height)
    return df

# Function to show Word documents in the sidebar
def show_word(file, width=300, height=200):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    html_content = f'<textarea style="width:{width}px;height:{height}px;">{text[:1000]}</textarea>'
    st.sidebar.markdown(html_content, unsafe_allow_html=True)
    return text

# Function to search for relevant information in the file content
def search_file_content(query, content):
    if isinstance(content, str):
        # Search for the query in the text content
        lines = content.split("\n")
        for line in lines:
            if query.lower() in line.lower():
                return f"Answer from file: {line.strip()}"
    elif isinstance(content, pd.DataFrame):
        # Search for the query in the DataFrame
        for col in content.columns:
            matches = content[content[col].astype(str).str.contains(query, case=False, na=False)]
            if not matches.empty:
                return f"Answer from file: {matches.head().to_string()}"
    return None

# Function to query Hugging Face's T5 model
def query_t5(query):
    # Add a prefix to ensure the response is in English
    response = generator(f"{query}", max_length=100, num_return_sequences=1)
    return response[0]["generated_text"]

# Sidebar file uploader
st.sidebar.title("File Space")
uploaded_file = st.sidebar.file_uploader(
    "Upload a file", type=["csv", "txt", "xlsx", "json", "pdf", "docx"]
)

# Initialize session state for storing conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the uploaded file
if uploaded_file is not None:
    st.sidebar.success(f"File {uploaded_file.name} uploaded successfully!")
    if uploaded_file.name.endswith(".csv"):
        content = show_data(uploaded_file, "csv")
    elif uploaded_file.name.endswith(".txt"):
        content = show_text(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        content = show_data(uploaded_file, "xlsx")
    elif uploaded_file.name.endswith(".json"):
        content = show_json(uploaded_file)
    elif uploaded_file.name.endswith(".pdf"):
        show_pdf(uploaded_file)
        content = "PDF content not searchable in this example."
    elif uploaded_file.name.endswith(".docx"):
        content = show_word(uploaded_file)
    else:
        st.warning("Unsupported file format.")
else:
    content = None

# Main Interface
st.title("Bot ~ V2")
with st.chat_message("assistant"):
    st.write("Hello there! How can I assist you today?")

# Display the conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):  # Use 'assistant' or 'user' as role
        st.markdown(message["content"])

# Input box for user to type their message
if user_input := st.chat_input("Type your message here..."):
    # Add user message to the conversation
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(f"User: {user_input}")

    # Generate bot response
    if content is not None:  # Check if content exists
        if isinstance(content, pd.DataFrame):
            if not content.empty:  # Check if DataFrame is not empty
                answer = search_file_content(user_input, content)
            else:
                answer = None
        else:
            answer = search_file_content(user_input, content)
    else:
        answer = None

    if answer:
        bot_response = answer
    else:
        bot_response = f"Answer from external resource: {query_t5(user_input)}"

    # Add bot message to the conversation
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    # Simulate chatbot "typing"
    with st.chat_message("assistant"):
        message_placeholder = st.empty()  # Placeholder for typing effect
        generated_text = ""
        for char in bot_response:
            generated_text += char  # Add the next character
            message_placeholder.text(generated_text)  # Update placeholder
            time.sleep(0.03)  # Adjust typing speed
        # Replace placeholder with final message
        message_placeholder.text(bot_response)

# Clear chat button
if st.button("Clear Chat", help="Click to clear the chat"):
    st.session_state.messages = []  # Clear the chat history
    st.rerun()