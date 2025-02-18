import streamlit as st
import os
from PyPDF2 import PdfReader
import docx
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message
from langchain.callbacks import get_openai_callback

openapi_key = st.secrets["OPENAI_API_KEY"]

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your file")
    st.header("Pakistan Organic Association")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf'], accept_multiple_files=True)
        openai_api_key = openapi_key
        process = st.button("Process")

    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        
        with st.spinner('Processing...'):
            files_text = get_files_text(uploaded_files)

            # Get text chunks
            text_chunks = get_text_chunks(files_text)
            st.write("File chunks created...")

            # Create vector store
            vectorstore = get_vectorstore(text_chunks)
            st.write("Vector store created...")

            # Create conversation chain
            st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)
            st.session_state.processComplete = True

    if st.session_state.processComplete:
        user_question = st.chat_input("Ask a question about your files.")
        if user_question:
            handle_user_input(user_question)

# Function to extract text from uploaded files
def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1]
        if file_extension == ".pdf":
            text += get_pdf_text(uploaded_file)
        elif file_extension == ".docx":
            text += get_docx_text(uploaded_file)
    return text

# Function to extract text from PDFs
def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = "".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return text

# Function to extract text from DOCX files
def get_docx_text(file):
    doc = docx.Document(file)
    return " ".join([para.text for para in doc.paragraphs])

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    return text_splitter.split_text(text)

# Function to create a vector store
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_texts(text_chunks, embeddings)

# Function to create the conversation chain
def get_conversation_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

# Function to handle user input
def handle_user_input(user_question):
    with get_openai_callback():
        response = st.session_state.conversation({'question': user_question})
    
    st.session_state.chat_history = response['chat_history']

    # Display chat messages
    response_container = st.container()
    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages.content, is_user=True, key=str(i))
            else:
                message(messages.content, key=str(i))

if __name__ == '__main__':
    main()
