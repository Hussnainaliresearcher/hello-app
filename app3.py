import streamlit as st
import os
from PyPDF2 import PdfReader
import docx
from langchain_community.chat_models import ChatOpenAI
from langchain.vectorstores import Qdrant
import random
from datetime import datetime
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
import string
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.docstore.document import Document
# Load secrets
openapi_key = st.secrets["OPENAI_API_KEY"]
qdrant_url = st.secrets["QDRANT_URL"]
qdrant_api_key = st.secrets["QDRANT_API_KEY"]
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

DATA_FOLDER = "data"
COLLECTION_NAME = "my_farm_files"  # Static name to reuse stored embeddings

def main():
    # Set page layout and title
    st.set_page_config(page_title="Q/A with your files", layout="centered")
    st.header("Retrieval QA Chain")

    # Initialize session states
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    # Display loading message only at the start
    if st.session_state.vectorstore is None:
        with st.empty():
            st.markdown("<h3 style='text-align: center;'>Loading...</h3>", unsafe_allow_html=True)
            with st.spinner("Checking stored data..."):
                initialize_vectorstore()

    # Show chat input when ready
    if st.session_state.processComplete:
        user_question = st.chat_input("Ask a question about your files.")
        if user_question:
            handle_user_input(user_question)

def initialize_vectorstore():
    """Loads vector store only once at startup."""
    vectorstore = load_existing_vectorstore(COLLECTION_NAME)

    if vectorstore:
        st.session_state.vectorstore = vectorstore
        st.session_state.conversation = get_qa_chain(vectorstore)
        st.session_state.processComplete = True
        st.success("Files loaded from Qdrant!")
    else:
        process_files()

def load_existing_vectorstore(collection_name):
    """Load stored vectorstore from Qdrant if available."""
    try:
        return Qdrant(
            embedding_function=embeddings,
            url=qdrant_url,
            prefer_grpc=True,
            api_key=qdrant_api_key,
            collection_name=collection_name
        )
    except Exception as e:
        return None  # No existing vectorstore found

def process_files():
    """Process and store files in Qdrant."""
    text_chunks_list = []
    pdf_files = [file for file in os.listdir(DATA_FOLDER) if file.endswith(".pdf")]

    if not pdf_files:
        st.error(f"No PDF files found in '{DATA_FOLDER}' folder.")
        return

    for pdf_file in pdf_files:
        file_path = os.path.join(DATA_FOLDER, pdf_file)
        file_text = get_pdf_text(file_path)
        text_chunks = get_text_chunks(file_text, pdf_file)
        text_chunks_list.extend(text_chunks)

    # Store embeddings in Qdrant
    vectorstore = store_vectorstore(text_chunks_list, COLLECTION_NAME)

    if vectorstore:
        st.session_state.vectorstore = vectorstore
        st.session_state.conversation = get_qa_chain(vectorstore)
        st.session_state.processComplete = True
        st.success("Files processed and stored in Qdrant!")

def get_pdf_text(pdf_path):
    """Extract text from a PDF file."""
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text

def get_text_chunks(text, filename):
    """Split text into chunks for embedding."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=80,
        chunk_overlap=20,
        length_function=len
    )
    return [Document(page_content=chunk, metadata={"source": filename}) for chunk in text_splitter.split_text(text)]

def store_vectorstore(text_chunks, collection_name):
    """Store document embeddings in Qdrant."""
    try:
        return Qdrant.from_documents(
            documents=text_chunks,
            embedding=embeddings,
            url=qdrant_url,
            prefer_grpc=True,
            api_key=qdrant_api_key,
            collection_name=collection_name,
        )
    except Exception as e:
        st.error(f"Error storing data in Qdrant: {e}")
        return None

def get_qa_chain(vectorstore):
    """Create a RetrievalQA chain."""
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
        return_source_documents=True
    )

def handle_user_input(user_question):
    """Handle user queries without reloading the vector store."""
    with st.spinner('Generating response...'):
        result = st.session_state.conversation({"query": user_question})
        response = result['result']
        source = result['source_documents'][0].metadata['source']

    st.session_state.chat_history.append(user_question)
    st.session_state.chat_history.append(f"{response} \n Source Document: {source}")

    response_container = st.container()
    with response_container:
        for i, message_text in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(message_text, is_user=True, key=str(i))
            else:
                message(message_text, key=str(i))

if __name__ == '__main__':
    main()
