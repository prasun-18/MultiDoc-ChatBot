import streamlit as st
import base64
import pandas as pd

# Create a sidebar
st.sidebar.title("file Space")
file = st.sidebar.file_uploader("Upload your pdf",type = "pdf")
st.sidebar.write("Preview of PDF.")

st.title("Bot-V1")
st.write("Chat Area.")







#########################################################################

# Function to display PDF
def show_pdf(pdf_file):
    pdf_data = pdf_file.read()
    pdf_base64 = base64.b64encode(pdf_data).decode("utf-8")
    pdf_embed_code = f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="300" height="200" type="application/pdf"></iframe>'
    st.sidebar.markdown(pdf_embed_code, unsafe_allow_html=True)


if file is not None:
    show_pdf(file)
