# 1. Import FastAPI and create app instance
from fastapi import FastAPI
app = FastAPI()

# 2. Import load_dotenv and load environment variables
from dotenv import load_dotenv
load_dotenv()  # Loads variables from .env file

# 3. Import os for environment variable access
import os

# 4. Import necessary LangChain modules (chains, embeddings, vectorstores)
from langchain.chains import RetrievalQA
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma


