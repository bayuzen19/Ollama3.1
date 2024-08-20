import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# Load the NVIDIA API key
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector embeddings

# Set up the Streamlit interface
st.title("Zen-Bot 🤖")


llm = ChatNVIDIA(model_name="meta/llama3-70b-instruct")


prompt_template = """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)


prompt1 = st.text_input("Enter your question from documents")


if st.button("Documents Embedding"):
    vector_embedding()
    st.success("Vector Store DB is ready! 🚀")


def display_message(message, is_user=True):
    if is_user:
        st.markdown(f"""
        <div style="text-align: right; margin-bottom: 10px;">
            <span style="background-color: #dcf8c6; padding: 10px; border-radius: 10px; display: inline-block;">
                {message}
            </span>
            <img src="https://img.icons8.com/ios-filled/50/000000/user-male-circle.png" alt="User" style="width: 40px; margin-left: 10px;">
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="text-align: left; margin-bottom: 10px;">
            <img src="https://img.icons8.com/ios-filled/50/000000/bot.png" alt="Bot" style="width: 40px; margin-right: 10px;">
            <span style="background-color: #ececec; padding: 10px; border-radius: 10px; display: inline-block;">
                {message}
            </span>
        </div>
        """, unsafe_allow_html=True)


if prompt1:
    display_message(prompt1, is_user=True) 
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)


    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    response_time = time.process_time() - start
    st.write(f"Response time: {response_time:.2f} seconds")


    display_message(response['answer'], is_user=False)


    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")


st.markdown(f"<div style='text-align: center; font-size: 12px; margin-top: 20px;'>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>", unsafe_allow_html=True)
