import streamlit as st
import pandas as pd
import json
import docx
import PyPDF2
from io import StringIO
import time

# Function to preview text files
def preview_txt(file):
    return file.read().decode("utf-8")

# Function to preview PDF files
def preview_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to preview DOCX files
def preview_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Function to preview JSON files
def preview_json(file):
    data = json.load(file)
    return json.dumps(data, indent=4)

# Function to preview Excel files
def preview_excel(file):
    df = pd.read_excel(file)
    return df.head().to_html()

# Streamlit UI setup
st.set_page_config(page_title="Bot", page_icon="🤖")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title
st.title("Bot 🤖")

# Sidebar for file upload
st.sidebar.title("Upload Files")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["txt", "pdf", "docx", "json", "xlsx"])

# Handle file upload in the sidebar
if uploaded_file is not None:
    st.sidebar.success(f"File '{uploaded_file.name}' uploaded successfully!")

    # Determine file type based on extension
    file_type = uploaded_file.name.split('.')[-1].lower()

    # Display preview based on file type
    st.sidebar.write(f"**Preview of {uploaded_file.name}:**")

    if file_type == "txt":
        st.sidebar.text_area("Text file preview", preview_txt(uploaded_file), height=200)
    elif file_type == "pdf":
        st.sidebar.text_area("PDF file preview", preview_pdf(uploaded_file), height=200)
    elif file_type == "docx":
        st.sidebar.text_area("DOCX file preview", preview_docx(uploaded_file), height=200)
    elif file_type == "json":
        st.sidebar.text_area("JSON file preview", preview_json(uploaded_file), height=200)
    elif file_type == "xlsx":
        st.sidebar.markdown(preview_excel(uploaded_file), unsafe_allow_html=True)
    else:
        st.sidebar.error("Unsupported file format")
else:
    st.sidebar.write("No file uploaded yet.")

# Display the greeting message from the bot
with st.chat_message("assistant"):
    st.write("Hello there! How can I assist you today?")

# Display chat history
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

    # Generate bot response (Echoing user's message)
    bot_response = f"Bot: {user_input}"

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

# Button to clear chat
if st.button("Clear Chat"):
    st.session_state.messages = []  # Reset the chat history
    st.rerun()  # Re-run the app to update the UI

