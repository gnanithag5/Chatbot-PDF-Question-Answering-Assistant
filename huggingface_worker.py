import os
import torch
import logging
from config import HUGGINGFACEHUB_API_TOKEN

# ----------------- LOGGER CONFIGURATION -----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# ----------------- LANGCHAIN IMPORTS -----------------
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub  # Modern import path

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
    """Initialize Hugging Face LLM and embedding model."""
    global llm_model, embedder

    log.info("Initializing Hugging Face model and embeddings...")

    # ✅ Load API key from config.py
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

    # Model repo configuration
    MODEL_REPO = "tiiuae/falcon-7b-instruct"

    # Model generation parameters
    generation_config = {
        "temperature": 0.1,
        "max_new_tokens": 600,
        "max_length": 600
    }

    # Initialize LLM
    llm_model = HuggingFaceHub(
        repo_id=MODEL_REPO,
        model_kwargs=generation_config
    )
    log.debug(f"Hugging Face model loaded: {MODEL_REPO}")

    # Initialize embeddings
    embedder = HuggingFaceInstructEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": DEVICE}
    )
    log.debug(f"Embedding model initialized on device: {DEVICE}")


# =========================================================
#                 DOCUMENT INGESTION PIPELINE
# =========================================================
def ingest_pdf(pdf_path: str):
    """Load a PDF, split it into chunks, and build the vector store."""
    global qa_engine

    log.info(f"Loading document from: {pdf_path}")
    pdf_loader = PyPDFLoader(pdf_path)
    pages = pdf_loader.load()
    log.debug(f"Document contains {len(pages)} page(s).")

    # Split document into smaller chunks
    chunker = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    doc_chunks = chunker.split_documents(pages)
    log.debug(f"Split into {len(doc_chunks)} chunks.")

    # Create Chroma vector store
    log.info("Creating Chroma vector database...")
    vectordb = Chroma.from_documents(doc_chunks, embedding=embedder)
    log.debug("Chroma index built successfully.")

    # Optional: inspect collections
    try:
        collections = vectordb._client.list_collections()
        log.debug(f"Chroma collections: {collections}")
    except Exception as ex:
        log.warning(f"Could not fetch Chroma collections: {ex}")

    # Build RetrievalQA pipeline
    qa_engine = RetrievalQA.from_chain_type(
        llm=llm_model,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25}),
        input_key="question",
        return_source_documents=False
    )
    log.info("RetrievalQA pipeline initialized successfully.")


# =========================================================
#                     PROMPT HANDLER
# =========================================================
def handle_query(user_input: str) -> str:
    """Generate an answer for a user’s query."""
    global qa_engine, conversation_memory

    log.info(f"Processing user query: {user_input}")

    response = qa_engine.invoke({
        "question": user_input,
        "chat_history": conversation_memory
    })

    answer = response["result"]
    log.debug(f"Model output: {answer}")

    conversation_memory.append((user_input, answer))
    log.debug(f"Conversation history length: {len(conversation_memory)}")

    return answer


# =========================================================
#                     ENTRY POINT
# =========================================================
if __name__ == "__main__":
    initialize_model()
    log.info("Hugging Face LLM and embeddings initialized successfully.")
