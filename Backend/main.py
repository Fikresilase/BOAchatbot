#import modules
from aiohttp import request
from fastapi import FastAPI
from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from fastapi import FastAPI
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from fastapi.concurrency import run_in_threadpool


# Run modules
load_dotenv()
app = FastAPI()

# Load the local embedding model
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")# Define the documents path
all_documents_path = "./documents"
all_documents = []

# Add all the PDF documents to a big list
for filename in os.listdir(all_documents_path):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(all_documents_path, filename))
        docs = loader.load()
        for i, doc in enumerate(docs):
            doc.metadata["source_file"] = filename
            doc.metadata["source_page"] = i + 1
        all_documents.extend(docs)

# Split the loaded documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs_chunks = text_splitter.split_documents(all_documents)

# Create a Chroma vector store using SentenceTransformer embeddings
vectordb = Chroma.from_documents(
    docs_chunks,
     embedding=embedding_model,        # pass embedding function, not precomputed
    persist_directory="./chroma_db"
)

# Persist the database to disk
vectordb.persist()

print(f"Loaded {len(all_documents)} documents, split into {len(docs_chunks)} chunks, and saved embeddings.")


# Create retrieval chain
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True  # <-- add this
)
# Define the request body for the POST request
class PromptRequest(BaseModel):
    prompt: str

# define my system prompt
system_prompt = "You are Mela, a helpful customer support assistant for Bank of Abyssinia trained by BoA AI center of excellence."

# combine the user and system prompts together
chat_prompt = f"{system_prompt}\nUser: {{query}}\nMela:"

#load all
print(f"Loaded {len(all_documents)} documents and split into {len(docs_chunks)} chunks.")
@app.post("/chat")
async def chat_with_docs(request: PromptRequest):
    # Combine system and user prompts
    chat_prompt = f"{system_prompt}\nUser: {request.prompt}\nMela:"

    # Call the retrieval chain with the combined prompt
    result = await run_in_threadpool(lambda: retrieval_chain.invoke({"query": chat_prompt}))
    
    answer = result["result"]
    sources = result["source_documents"]
    
    return {
        "answer": answer,
        "sources": [
            {
                "text": doc.page_content,
                "file": doc.metadata.get("source_file"),
                "page": doc.metadata.get("source_page")
            }
            for doc in sources
        ]
    }

