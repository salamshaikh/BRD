import streamlit as st
import pdfplumber
import docx2txt
import os
import io
import shutil
import time
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import openai

# Load environment variables from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Streamlit page config
st.set_page_config(page_title="📄 BRD Chat Assistant")
st.title("📄 BRD Chat Assistant")

# Load PDF text using pdfplumber
def load_pdf_text(file_bytes):
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Load DOCX text using docx2txt
def load_docx_text(file_path):
    return docx2txt.process(file_path)

# Clean and create vector store from text
def create_vector_store(text, persist_directory="vector_db"):
    # Safely clean old vector DB if it exists
    if os.path.exists(persist_directory):
        try:
            vectordb = Chroma(
                persist_directory=persist_directory,
                embedding_function=OpenAIEmbeddings()
            )
            vectordb.delete_collection()
        except Exception as e:
            print("Warning during DB cleanup:", e)
        time.sleep(1)
        shutil.rmtree(persist_directory, ignore_errors=True)

    # Create fresh vector DB
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents([text])
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb

# Load vector DB and setup QA chain
def get_qa_chain(persist_directory="vector_db"):
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=OpenAIEmbeddings()
    )
    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# File uploader
uploaded_file = st.file_uploader("Upload BRD file (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file:
    with st.spinner("Reading and indexing document..."):
        if uploaded_file.name.endswith(".pdf"):
            text = load_pdf_text(uploaded_file.read())
        else:
            temp_path = f"temp_data/{uploaded_file.name}"
            os.makedirs("temp_data", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            text = load_docx_text(temp_path)

        create_vector_store(text)
        st.success("✅ Document indexed successfully! Ask your questions below.")

    # Text input for user queries
    query = st.text_input("Ask a question about the BRD:")
    if query:
        with st.spinner("Thinking..."):
            try:
                qa_chain = get_qa_chain()
                response = qa_chain({"query": query})
                st.markdown("### ✅ Answer:")
                st.write(response["result"])
            except Exception as e:
                st.error("❌ Something went wrong while processing your query.")
                st.exception(e)

# Optional manual clear button
if st.button("🔄 Clear vector database manually"):
    if os.path.exists("vector_db"):
        shutil.rmtree("vector_db", ignore_errors=True)
        st.success("Vector DB cleared successfully.")
