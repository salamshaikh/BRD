import streamlit as st
from dotenv import load_dotenv
import fitz  # PyMuPDF
import os
import docx2txt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load PDF text
def load_pdf_text(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Load DOCX text
def load_docx_text(file_path):
    return docx2txt.process(file_path)

# Create Vector Store
def create_vector_store(text, persist_directory="vector_db"):
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

# Load Vector DB and setup QA chain
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

# Streamlit UI
st.set_page_config(page_title="📄 BRD Chat Assistant")
st.title("📄 BRD Chat Assistant")

uploaded_file = st.file_uploader("Upload BRD file (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file:
    file_path = f"temp_data/{uploaded_file.name}"
    os.makedirs("temp_data", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load text based on file type
    if uploaded_file.name.endswith(".pdf"):
        text = load_pdf_text(file_path)
    else:
        text = load_docx_text(file_path)

    with st.spinner("Indexing document..."):
        create_vector_store(text)
        st.success("Document indexed successfully!")

    query = st.text_input("Ask a question about the BRD:")

    if query:
        qa_chain = get_qa_chain()
        with st.spinner("Searching..."):
            response = qa_chain({"query": query})
        st.markdown("### ✅ Answer:")
        st.write(response["result"])
