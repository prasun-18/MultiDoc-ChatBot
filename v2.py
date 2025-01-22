# pip install streamlit pandas PyPDF2 python-docx openpyxl

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

st.title("Bot-V2")

# main data variable -> uploaded_file 


