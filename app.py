import(‘pysqlite3’)
import sys
sys.modules[‘sqlite3’] = sys.modules.pop(‘pysqlite3’)

import streamlit as st
from src.Bot.utils import OCR
import time
import os
import gc
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import shutil
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import SelfQueryRetriever
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains.retrieval import create_retrieval_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.title("Conversational RAG with PDF uploads, OCR, and Chat History")
st.write("Upload PDFs, perform OCR, and chat with their content")

st.header("Enter API Keys")
if "groq_api_key" not in st.session_state:
    st.session_state["groq_api_key"] = None
if "hf_token" not in st.session_state:
    st.session_state["hf_token"] = None

if "pdf" not in st.session_state:
    st.session_state["pdf"] = False

if "chat_button" not in st.session_state:
    st.session_state["chat_button"] = False

if "default_question" not in st.session_state:
    st.session_state["default_question"] = False

if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

# Input for GROQ API and Hugging Face API
groq_api_key = st.text_input("Enter your GROQ API Key", type="password")
hf_token = st.text_input("Enter your Hugging Face API Key", type="password")

if st.button("Submit API Keys"):
    st.session_state["groq_api_key"] = groq_api_key
    st.session_state["hf_token"] = hf_token
    st.success("API keys submitted successfully!")

if st.session_state["groq_api_key"] and st.session_state["hf_token"]:
    os.environ["GROQ_API_KEY"] = st.session_state["groq_api_key"]
    os.environ['HF_TOKEN'] = st.session_state["hf_token"]

    llm = ChatGroq(groq_api_key=st.session_state["groq_api_key"], model_name="Gemma2-9b-It")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    st.write("API Keys are set. You can now upload a PDF and start working.")

    file_upload = st.sidebar.file_uploader("Upload your PDF", type="pdf")

    if file_upload:
        input_pdf_path = os.path.join(os.getcwd(), "uploaded_file.pdf")
        with open(input_pdf_path, "wb") as f:
            f.write(file_upload.getvalue())

    ocr_button = st.sidebar.button("OCR")
    if ocr_button:
        ocr = OCR(input_pdf_path)
        output_file_path = ocr.do_ocr()

        st.session_state.pdf = True
        st.write(output_file_path)

        # Clear existing Chroma DB instance
        #st.session_state.vectorstore = None

    chat_button = st.sidebar.button("Chat")

    if chat_button:
        st.session_state.chat_button = True

    clear_history = st.sidebar.button("Clear History")

    if clear_history:
        
        st.session_state.vectorstore = None  # Ensure Chroma DB is cleared
        st.session_state["pdf"] = False
        st.session_state["chat_button"] = False
        st.session_state["default_question"] = False
        st.session_state["vectorstore"] = None
        st.write("History cleared and Chroma DB removed from memory.")

    if st.session_state.pdf and st.session_state.chat_button:
        default_output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(default_output_dir, exist_ok=True)
        output = os.path.join(default_output_dir, "output.pdf")
        loader = PyMuPDFLoader(output)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)

        # Create a new Chroma instance
        st.session_state.vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = st.session_state.vectorstore.as_retriever()

        system_prompt = (
            """You are an intelligent assistant tasked with extracting specific details from a document. I will list the fields I need information on, and you should provide the answers based on the document content. Please extract and format the answers clearly for each of the following fields:

            Please provide the answers in the same order, clearly labeling each field.
                "\n\n"
                "{context}"
            """
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        if "default_question" not in st.session_state:
            st.session_state.default_question = False

        default_questions = st.button("Ask Default Questions to PDF")

        question = ["who is second party give its details",
                    "who is first party", "carpet Area and builtup area", "Rent escalation details",
                    "Transaction Type (sale/ Lease)", "Registry Date", "Document or registration Number",
                    "Village name", "Transaction Based on (Builtup or Carpet if both are given consider Builtup)",
                    "Stamp Duty if Given", "Total Car Parking", "Refund of Interest Fee",
                    "escalation chart with start date, end date, rent per sqft after escalation percentage by calculation",
                    "Car parking Charges", "Cam charges per Square feet", "rent per in 1st year",
                    "rent value", "Lease start date", "Lease End date calculate by start date if not given",
                    "lock in period in months if given if not give 'NA'", "Notice Period in days or months",
                    "Location and Floor of area leased to second party", "security Deposit amount"]

        if default_questions:
            resp = []
            for i in question:
                response = rag_chain.invoke({"input": i})
                if 'answer' in response:
                    resp.append(f"{i} : {response['answer']}")
                else:
                    resp.append(f"{i} : No answer found")
                time.sleep(1)
            st.write("Default questions are selected")
            st.write(resp)

        user_input = st.text_input("Your question:")
        if user_input:
            response = rag_chain.invoke({"input": user_input})
            st.write("Assistant:", response['answer'])

    else:
        st.write("<--- Run OCR FIRST")
else:
    st.warning("Please enter your API keys to proceed.")
