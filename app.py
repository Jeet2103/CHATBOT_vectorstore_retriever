import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
import tempfile

from dotenv import load_dotenv
load_dotenv()

## Load the GROQ API Key
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

## If you do not have open AI key use the below Huggingface embedding
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-70b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question: {input}
    """
)

def create_vector_embedding(uploaded_file):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_file_path = tmp_file.name
        
        st.session_state.loader = PyPDFLoader(temp_file_path)  # Data Ingestion step using temp file path
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        
        # Clean up the temporary file
        os.remove(temp_file_path)

st.title("RAG Document Q&A With Groq And Lama3")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")  # Allow users to upload a PDF file

if uploaded_file and st.button("Create Document Embedding"):
    create_vector_embedding(uploaded_file)
    st.write("Vector Database is ready")

# Text input followed by the submit button
user_prompt = st.text_input("Enter your query from the research paper")

submit_button = st.button("Submit")

import time

if submit_button and user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_prompt})
    print(f"Response time : {time.process_time() - start}")

    st.write(response['answer'])

    ## With a Streamlit expander
    with st.expander("Document similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')
