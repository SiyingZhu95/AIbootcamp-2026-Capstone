import os
import logging
from typing import Optional
import streamlit as st

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "./data"
VECTOR_STORE_PATH = "./vectorstore/silver_support_chroma"
EMBEDDING_MODEL = "models/gemini-embedding-001"


def get_api_key() -> str:
    """
    Retrieves the Google Gemini API key securely from Streamlit secrets.
    Falls back to environment variables if running outside Streamlit context.
    """
    api_key = None

    # Check Streamlit secrets
    try:
        if "API_KEY" in st.secrets:
            api_key = st.secrets["API_KEY"]
        elif "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception as e:
        logger.debug(f"Streamlit secrets not accessible directly: {e}")

    # Fallback to environment variable
    if not api_key:
        api_key = os.getenv("API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "API key missing. Please define 'API_KEY' inside '.streamlit/secrets.toml' "
            "or set the 'API_KEY' environment variable."
        )

    return api_key


def ingest_documents() -> Optional[Chroma]:
    """
    Loads PDF and TXT policy documents from DATA_DIR, splits them into semantic 
    chunks, generates Gemini embeddings, and persists them into a Chroma vector DB.

    Returns:
        Optional[Chroma]: The populated Chroma vector store instance, or None if no files exist.
    """
    logger.info(f"Starting ingestion workflow from source directory: '{DATA_DIR}'")

    # Ensure source directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        logger.warning(f"Created missing directory '{DATA_DIR}'. Please add PDF/TXT policy files and re-run.")
        print(f"Warning: Directory '{DATA_DIR}' was missing. Created it now. Please add documents.")
        return None

    # Retrieve API key
    api_key = get_api_key()

    # 1. Document Loading
    documents = []

    # Load PDF documents
    try:
        pdf_loader = DirectoryLoader(
            DATA_DIR,
            glob="*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        pdf_docs = pdf_loader.load()
        documents.extend(pdf_docs)
        logger.info(f"Loaded {len(pdf_docs)} PDF page document(s).")
    except Exception as e:
        logger.error(f"Error loading PDF documents: {e}")

    # Load TXT documents
    try:
        txt_loader = DirectoryLoader(
            DATA_DIR,
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True
        )
        txt_docs = txt_loader.load()
        documents.extend(txt_docs)
        logger.info(f"Loaded {len(txt_docs)} TXT document(s).")
    except Exception as e:
        logger.error(f"Error loading TXT documents: {e}")

    # Handle empty document directory
    if not documents:
        msg = f"No valid PDF or TXT files found in '{DATA_DIR}'. Ingestion aborted."
        logger.warning(msg)
        print(f"Warning: {msg}")
        return None

    logger.info(f"Total raw document objects loaded: {len(documents)}")

    # 2. Text Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=60,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    total_chunks = len(chunks)
    logger.info(f"Successfully split documents into {total_chunks} chunks.")
    print(f"Total chunks created: {total_chunks}")

    # 3. Embedding Initialization & Vector Database Persistence
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=api_key
        )

        logger.info(f"Persisting vectors to Chroma store at: '{VECTOR_STORE_PATH}'")

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=VECTOR_STORE_PATH
        )

        logger.info("Chroma vector store successfully created and persisted.")
        return vector_store

    except Exception as e:
        logger.error(f"Failed during embedding generation or database creation: {e}")
        raise e


if __name__ == "__main__":
    print("==================================================")
    print("  Silver Support Scheme - Document Ingestion Process  ")
    print("==================================================")

    try:
        vector_db = ingest_documents()
        if vector_db:
            print("\n✅ Ingestion completed successfully!")
            print(f"📁 Vector store persisted at: {VECTOR_STORE_PATH}")
        else:
            print("\n⚠️ Ingestion halted: No documents were ingested.")
    except Exception as err:
        print(f"\n❌ Ingestion failed with error: {err}")