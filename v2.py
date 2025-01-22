# pip install streamlit pandas PyPDF2 python-docx openpyxl
import time
import streamlit as st
import pandas as pd
import json
import base64
from PyPDF2 import PdfReader
from docx import Document
##############################################

# Sidebar

st.sidebar.title("File Space")


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

# Sidebar file uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload a file", type=["csv", "txt", "xlsx", "json", "pdf", "docx"]
)

# Display the uploaded file
if uploaded_file is not None:
    st.sidebar.success(f"File {uploaded_file.name} uploaded successfully!")
    #st.write("Contents of the uploaded file:")

    # Determine the file type and process accordingly
    if uploaded_file.name.endswith(".csv"):
        data = show_data(uploaded_file, "csv")
        #st.write(data)
    elif uploaded_file.name.endswith(".txt"):
        content = show_text(uploaded_file)
        #st.text(content)
    elif uploaded_file.name.endswith(".xlsx"):
        data = show_data(uploaded_file, "xlsx")
        #st.write(data)
    elif uploaded_file.name.endswith(".json"):
        content = show_json(uploaded_file)
        st.json(content)
    elif uploaded_file.name.endswith(".pdf"):
        show_pdf(uploaded_file)
        # pdf_reader = PdfReader(uploaded_file)
        # text = ""
        # for page in pdf_reader.pages:
        #     text += page.extract_text()
        # st.text(text)
    elif uploaded_file.name.endswith(".docx"):
        content = show_word(uploaded_file)
        #st.text(content)
    else:
        st.warning("Unsupported file format.")

##########################################################

# Main Interface 

# main data variable -> uploaded_file 



st.title("Bot-V2")



with st.chat_message("assistant"):
    st.write("Hello there! How can I assist you today?")
        
# Initialize session state for storing conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []  # List to store chat history

# Display the conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):  # Use 'assistant' or 'user' as role
        st.markdown(message["content"])

# Input box for user to type their message (((Bot response)))
if user_input := st.chat_input("Type your message here..."):
    # Add user message to the conversation
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(f"User: {user_input}")
#########################################################################

    # Generate bot response (Replace with your chatbot logic)
    bot_response = f"Bot: {user_input}"  
    
#########################################################################
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







