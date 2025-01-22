import streamlit as st
import base64
import pandas as pd
import time

# Create a sidebar
st.sidebar.title("file Space")
file = st.sidebar.file_uploader("Upload your pdf",type = "pdf")
st.sidebar.write("Preview of PDF.")

st.title("Bot-V1")
eraser = st.empty()
eraser.write("Upload the PDF to turn on the ChatBot")

#########################################################################
#  load the Model and implement Main logic here.
#########################################################################


with st.chat_message("assistant"):
    st.write("Hello there! How can I assist you today?")
        
# Initialize session state for storing conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []  # List to store chat history

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





# Function to display PDF
def show_pdf(pdf_file):
    pdf_data = pdf_file.read()
    pdf_base64 = base64.b64encode(pdf_data).decode("utf-8")
    pdf_embed_code = f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="300" height="200" type="application/pdf"></iframe>'
    st.sidebar.markdown(pdf_embed_code, unsafe_allow_html=True)












if file is not None:
    show_pdf(file)
