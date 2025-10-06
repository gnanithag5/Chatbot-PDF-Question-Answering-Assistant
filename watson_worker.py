import os
import torch
import logging
from config import (
    WATSONX_API_KEY,
    WATSONX_PROJECT_ID,
    WATSONX_URL
)

# ----------------- LOGGER CONFIGURATION -----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# ----------------- LANGCHAIN IMPORTS -----------------
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_ibm import WatsonxLLM
from langchain_core.prompts import PromptTemplate

# ----------------- DEVICE CHECK -----------------
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ----------------- GLOBAL OBJECTS -----------------
qa_engine = None
conversation_memory = []
llm_model = None
embedder = None


# =========================================================
#               MODEL & EMBEDDING INITIALIZATION
# =========================================================
def initialize_model():
    """Initialize IBM Watsonx LLM and embedding model."""
    global llm_model, embedder

    log.info("Setting up IBM Watsonx LLM and embedding model...")

    # IBM Watsonx model configuration
    MODEL_PATH = "meta-llama/llama-3-3-70b-instruct"

    # Local model parameters
    max_new_tokens = 256
    temperature = 0.1

    # Create LLM object
    llm_model = WatsonxLLM(
        model_id=MODEL_PATH,
        url=WATSONX_URL,
        project_id=WATSONX_PROJECT_ID,
        params={
            "max_new_tokens": max_new_tokens,
            "temperature": temperature
        },
        apikey=WATSONX_API_KEY
    )
    log.debug("Watsonx LLM successfully configured.")

    # Create embedding model using Hugging Face
    embedder = HuggingFaceInstructEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": DEVICE}
    )
    log.debug(f"Embedding model loaded on device: {DEVICE}")


# =========================================================
#                 DOCUMENT INGESTION PIPELINE
# =========================================================
def ingest_pdf(pdf_path: str):
    """Load a PDF, split into chunks, and index with embeddings."""
    global qa_engine

    log.info(f"Reading PDF from: {pdf_path}")
    pdf_loader = PyPDFLoader(pdf_path)
    pages = pdf_loader.load()
    log.debug(f"Extracted {len(pages)} page(s) from document.")

    # Split into smaller overlapping chunks
    chunker = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    doc_chunks = chunker.split_documents(pages)
    log.debug(f"Document divided into {len(doc_chunks)} chunks for embedding.")

    # Create Chroma vector store
    log.info("Building Chroma index...")
    vectordb = Chroma.from_documents(doc_chunks, embedding=embedder)
    log.debug("Chroma index created successfully.")

    # Optional: log Chroma collections (not required)
    try:
        collections = vectordb._client.list_collections()
        log.debug(f"Chroma collections: {collections}")
    except Exception as ex:
        log.warning(f"Could not fetch Chroma collections: {ex}")

    # Assemble RetrievalQA chain for question answering
    qa_engine = RetrievalQA.from_chain_type(
        llm=llm_model,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25}),
        input_key="question",
        return_source_documents=False
    )
    log.info("RetrievalQA pipeline initialized.")


# =========================================================
#                     PROMPT HANDLER
# =========================================================
def handle_query(user_input: str) -> str:
    """Generate a contextual response for user query."""
    global qa_engine, conversation_memory

    log.info(f"Received query: {user_input}")

    # Run the retrieval chain
    response_data = qa_engine.invoke({
        "question": user_input,
        "chat_history": conversation_memory
    })

    answer_text = response_data["result"]
    log.debug(f"Response generated: {answer_text}")

    # Store chat history
    conversation_memory.append((user_input, answer_text))
    log.debug(f"Conversation history size: {len(conversation_memory)}")

    return answer_text


# =========================================================
#                     ENTRY POINT
# =========================================================
if __name__ == "__main__":
    initialize_model()
    log.info("Watsonx LLM and embeddings are ready.")
