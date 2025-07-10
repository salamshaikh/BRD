import streamlit as st
from dotenv import load_dotenv
import os
import io
import pdfplumber
import docx2txt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import openai
import shutil

# Load API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

st.set_page_config(page_title="📄 BRD Chat Assistant")
st.title("📄 BRD Chat Assistant")

# --- Load PDF text using pdfplumber ---
def load_pdf_text(file_bytes):
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# --- Load DOCX text ---
def load_docx_text(file_path):
    return docx2txt.process(file_path)

# --- Create Vector Store ---
def create_vector_store(text, persist_directory="vector_db"):
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([text])
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()
    return vectordb

# --- Get QA Chain ---
def get_qa_chain(persist_directory="vector_db"):
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())
    retriever = vectordb.as_retriever()
    return RetrievalQA.from_chain_type(llm=OpenAI(temperature=0), chain_type="stuff", retriever=retriever, return_source_documents=True)

# --- File Upload UI ---
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
        st.success("Document indexed successfully!")

    query = st.text_input("Ask a question about the BRD:")
    if query:
        with st.spinner("Thinking..."):
            try:
                qa_chain = get_qa_chain()
                response = qa_chain({"query": query})
                st.markdown("### ✅ Answer:")
                st.write(response["result"])
            except Exception as e:
                st.error("Something went wrong during question answering.")
                st.exception(e)
