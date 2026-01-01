from flask import Flask, render_template, request
import os
import requests
from prompts import contextualize_q_system_prompt, qa_system_prompt

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.embeddings import HuggingFaceEmbeddings
from langchain_classic.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS

app = Flask(__name__)


chat_history = {}
FAISS_DIR = "faiss"

#Loading Multiple pdfs using Directory Loader
def get_documents():
    loader = DirectoryLoader('data_pdf', loader_cls=PyPDFLoader, glob="**/*.pdf", show_progress=True)
    docs = loader.load()
    return docs

#Chunking
def get_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chubk_overlap=100, length_function=len)
    chunks = text_splitter.split_documents(docs)
    return chunks

#creating embeddings and saving it locally
def get_embeddings():
    FAISS_PATH = os.path.join(os.getcwd(), FAISS_DIR)
    documents = get_documents()
    chunks = get_chunks(documents)

    if os.path.exists(FAISS_PATH):
        print("Index exists. Loading from {path}")
        db = FAISS.load_local(chunks, HuggingFaceEmbeddings(), allow_dangerous_deserialization=True)
    else:
        print("Index doesn't exits, hence creating one")

        db = FAISS.from_documents(chunks, HuggingFaceEmbeddings())
        db.save_local(FAISS_PATH)

    return db

def get_retriever():
    db = get_embeddings()
    retriever = db.as_retriever()
    return retriever

llm = ChatOllama(model="llama3.1:8b")

# Hitsory aware retriever
context_q_prompt = ChatPromptTemplate.from_messages([("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human","{question}")])
history_aware_retriever = create_history_aware_retriever(llm, get_retriever(), context_q_prompt)

qa_prompt = ChatPromptTemplate.from_messages([("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{question}")])

# Stuffing chain type, creates the Answerer 
qa_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)



@app.route('/')
def home_page():
    return render_template("index.html")

@app.route('/chat', methods= ["GET", "POST"])
def chat_page():
    if request.method == "GET":
        return render_template("chat.html")
    
    if request.method == "POST":
        data = request.json

        user_input = data.get("msg")
        

if __name__ == "__main__":
    app.run(debug=True)


