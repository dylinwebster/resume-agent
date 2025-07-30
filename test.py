# resume_agent.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
print("API key loaded:", bool(openai_key))  # This prints True/False

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

# Load your resume or case study PDF
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

def load_all_documents(folder_path):
    docs = []
    folder = Path(folder_path)
    for file_path in folder.glob("*"):
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif ext == ".docx":
            loader = Docx2txtLoader(str(file_path))
        elif ext in [".txt", ".md"]:
            loader = TextLoader(str(file_path))
        else:
            print(f"Skipping unsupported file: {file_path.name}")
            continue
        docs.extend(loader.load())
    return docs

# Load all documents from the /docs folder (must be inside the AGENT directory)
docs = load_all_documents("docs")

# Split text into chunks for embedding
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = splitter.split_documents(docs)

# Embed and store in vector DB
embedding = OpenAIEmbeddings(openai_api_key=openai_key)
vectordb = Chroma.from_documents(split_docs, embedding)

# Create retriever and LLM-powered QA chain
retriever = vectordb.as_retriever()
llm = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-4")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Interactive loop
print("Your resume agent is ready. Type your question or 'quit' to exit.")
while True:
    query = input("\nAsk something: ")
    if query.lower() in ["exit", "quit"]:
        break
    result = qa_chain.run(query)
    print("\n" + result)
