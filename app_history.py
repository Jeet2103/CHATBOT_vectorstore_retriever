import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
# from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = st.secrets['GROQ_API_KEY']
HF_TOKEN = st.secrets['HF_TOKEN']

## Load the GROQ API Key
# os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['GROQ_API_KEY'] = GROQ_API_KEY
# groq_api_key = os.getenv("GROQ_API_KEY")

## If you do not have open AI key use the below Huggingface embedding
# os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
os.environ['HF_TOKEN'] = HF_TOKEN
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set streamlit app
st.title("Conversational RAG with PDF Uploads and Chat History")
st.write("Uploads Pdf's and chat with their content")

# Model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-70b-8192")

# Chat interface
session_id = st.text_input("Session ID", value= "Default_session")

if 'store' not in st.session_state:
    st.session_state.store = {}


uploaded_files = st.file_uploader("Choose  PDF's file", type= "pdf", accept_multiple_files= True)

##Process Upload PDF's

if uploaded_files:
    documents=[]
    for uploaded_file in uploaded_files:
        temppdf=f"./temp.pdf"
        with open(temppdf,"wb") as file:
            file.write(uploaded_file.getvalue())
            file_name=uploaded_file.name

        loader=PyPDFLoader(temppdf)
        docs=loader.load()
        documents.extend(docs)

        #remove the temp.pdf
        os.remove(temppdf)
    # Split and create embeddings for the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    contextualize_q_system_prompt=(
        "Given a chat history and the latest user question"
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)


    ## Answer question

    # Answer question
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
    rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)
    def get_session_history(session:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]=ChatMessageHistory()
        return st.session_state.store[session_id]
    
    conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
    
    user_input = st.chat_input(placeholder="Your question:")
    # submit_button = st.button("Submit")
    if user_input : #and submit_button:
        session_history=get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id":session_id}
            },  # constructs a key "abc123" in `store`.
        )
        # st.write(st.session_state.store)
        st.write("Assistant:", response['answer'])
        st.write("Chat History:", session_history.messages)
else:
    st.warning("Please upload the files first.")