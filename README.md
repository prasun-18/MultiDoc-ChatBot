# MultiDoc-ChatBot 🤖

MultiDoc-ChatBot is a Streamlit-based chatbot that allows users to upload multiple file formats (PDF, DOCX, CSV, JSON, TXT, XLSX) and interact with their contents using an AI-powered retrieval-based QA system. The chatbot leverages LangChain, FAISS for vector storage, and Hugging Face embeddings for natural language understanding.

## Features
- 📂 **Supports multiple file formats**: PDF, DOCX, CSV, JSON, TXT, XLSX
- 🔍 **Processes and extracts text from files**
- 🧠 **Embeds text using Hugging Face embeddings**
- 🔎 **Retrieves answers from document content using FAISS**
- 💬 **Provides a chatbot interface with AI-powered responses**
- 🎭 **Handles both document-based QA and general chatbot queries**

## Installation

### Prerequisites
Ensure you have Python installed (Python 3.8 or higher is recommended).

### Clone the Repository
```sh
git clone https://github.com/prasun-18/MultiDoc-ChatBot.git
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

## Usage

### Running the Streamlit App locally
```sh
streamlit run MultiDoc-ChatBot.py
```

### To run into google colab use these 3 command insted
```
! pip install streamlit -q
```
```
!wget -q -O - ipv4.icanhazip.com
```
```
!streamlit run MultiDoc-ChatBot.py & npx localtunnel --port 8501
```

### Uploading a File
1. Open the Streamlit web interface.
2. Use the sidebar to upload a file (PDF, DOCX, CSV, JSON, TXT, XLSX).
3. The system will process the file and store its content in a vector database.

### Asking Questions
- Type your query in the chat input.
- If a document is uploaded, the chatbot retrieves relevant answers based on the content.
- If no document is uploaded, the chatbot provides general AI-based responses.

### Clearing Chat
Click the **Clear Chat** button to reset the conversation.

## File Processing Details

| File Type | Processing Method |
|-----------|------------------|
| **PDF**   | Extracts text using PyPDF2 |
| **DOCX**  | Extracts text from paragraphs using python-docx |
| **CSV/XLSX** | Reads tabular data using pandas |
| **JSON**  | Formats and extracts JSON content |
| **TXT**   | Reads plain text files |

## Technologies Used
- **Streamlit**: UI framework for the chatbot.
- **LangChain**: For text embedding and retrieval-based QA.
- **FAISS**: Vector storage for efficient similarity searches.
- **Hugging Face Embeddings**: For sentence embeddings.
- **PyPDF2, python-docx, pandas, json**: For file parsing and text extraction.

## Environment Variables
To use the Hugging Face model, set up your API token:
```sh
export HUGGINGFACEHUB_API_TOKEN='your_api_token_here'
```
Or add it to your `.env` file.

### For better compatibility with API calls over Google Colab, I recommend using a model like `tiiuae/falcon-7b-instruct` or `bigscience/bloomz-7b1`, both of which are well-suited for answering questions and are optimized for inference tasks via the Hugging Face Inference API.

## Limitations

- Here I used `tiiuae/falcon-7b-instruct` for question-answers(due to lack of resources).
- These models `tiiuae/falcon-7b-instruct` or `bigscience/bloomz-7b1` are best suited for low ram and storage (Ram-4GB, storage<=4GB)
- Token limitations : `inputs tokens` + `max_new_tokens` must be <= 8192.

## Future Enhancements
- [ ] Support for additional file formats.
- [ ] Improved document segmentation techniques.
- [ ] Enhanced UI/UX with chat history and better file previews.

---
### Here are some other small models you can explore and experiment with
- intfloat/e5-small
- all-MiniLM-L6-v2 (compact, fast, and effective)
- multi-qa-MiniLM-L6-cos-v1 (fine-tuned for question answering and semantic search)
- all-MPNet-base-v2 (higher accuracy with a larger model size)

---

#### Thanks for reading till the end. XD

### Author
Developed by **Prasun Kumar** 🚀

