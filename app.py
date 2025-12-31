from flask import Flask, render_template, request
import os
import requests

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.embeddings import HuggingFaceEmbeddings
from langchain_classic.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama

app = Flask(__name__)

llm = ChatOllama(model="llama3.1:8b")

@app.route('/')
def home_page():
    return render_template("index.html")

@app.route('/chat', methods= ["GET", "POST"])
def chat_page():
    if request.method == "GET":
        return render_template("chat.html")

if __name__ == "__main__":
    app.run(debug=True)


