# Step 1: Install Required Libraries
# Run this in your terminal before executing the script:
# pip install langchain huggingface_hub PyPDF2 python-docx pandas openai faiss-cpu langchain-huggingface

# Step 2: Import Required Libraries
import os
import PyPDF2
import docx
import pandas as pd
import json
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint

# Step 3: Define Functions to Read Different File Types
def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text

def read_word(file_path):
    doc = docx.Document(file_path)
    text = '\n'.join([para.text for para in doc.paragraphs])
    return text

def read_excel(file_path):
    df = pd.read_excel(file_path)
    return df.to_string()

def read_csv(file_path):
    # Read CSV file as a DataFrame
    df = pd.read_csv(file_path)
    # Convert the DataFrame to a single string for processing
    return df.to_string()

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return json.dumps(data)

def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Step 4: Load and Process Files
def load_and_process_file(file_path):
    if file_path.endswith('.pdf'):
        text = read_pdf(file_path)
    elif file_path.endswith('.docx'):
        text = read_word(file_path)
    elif file_path.endswith('.xlsx'):
        text = read_excel(file_path)
    elif file_path.endswith('.csv'):
        text = read_csv(file_path)  # Handle CSV files
    elif file_path.endswith('.json'):
        text = read_json(file_path)
    elif file_path.endswith('.txt'):
        text = read_txt(file_path)  # Handle TXT files
    else:
        raise ValueError("Unsupported file type")

    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    
    return texts

# Step 5: Create Embeddings and Vector Store
def create_vector_store(texts):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")  # Explicitly pass model_name
    vector_store = FAISS.from_texts(texts, embeddings)
    return vector_store

# Step 6: Set Up the Question-Answering System
def setup_qa_system(vector_store):
    # Provide your Hugging Face API token here
    huggingface_api_token = "Replace with your actual API token"  # Replace with your actual API token

    # Use HuggingFaceEndpoint with the Falcon-7B-Instruct model
    llm = HuggingFaceEndpoint(
        repo_id="tiiuae/falcon-7b-instruct",  # Use a lightweight and instruction-tuned model
        huggingfacehub_api_token=huggingface_api_token,
        temperature=0.2,  # Adjust temperature for better instruction-following
    )
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())
    return qa

# Step 7: Main Function to Interact with User
def main():
    # Prompt user for file path
    file_path = input("Enter the path to your file: ")
    
    # Load and process the file
    try:
        texts = load_and_process_file(file_path)
        print("File loaded and processed successfully!")
    except Exception as e:
        print(f"Error processing file: {e}")
        return

    # Create vector store
    vector_store = create_vector_store(texts)
    print("Vector store created!")

    # Set up QA system
    try:
        qa_system = setup_qa_system(vector_store)
        print("Bot's Question-Answering system is ready!")
    except Exception as e:
        print(f"Error setting up QA system: {e}")
        return

    # Interactive Q&A loop
    while True:
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        try:
            # Retrieve answer from the vector store
            retrieval_result = vector_store.similarity_search_with_score(question, k=1)

            # Check if the answer is from the file
            if retrieval_result and retrieval_result[0][1] < 0.5:  # Threshold for similarity
                print(f"Answer from the file: {retrieval_result[0][0].page_content}")
            else:
                # Use LLM to provide the answer if not found in the file
                answer = qa_system.invoke(question)
                print("Answer is not present in the provided file.")
                print(f"Answer (from model): {answer}")
        except Exception as e:
            print(f"Error answering question: {e}")

# Step 8: Run the Application
if __name__ == "__main__":
    main()
