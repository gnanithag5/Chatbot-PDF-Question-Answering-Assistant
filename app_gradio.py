import os
import torch
import logging
import gradio as gr
from config import (
    WATSONX_API_KEY,
    WATSONX_PROJECT_ID,
    WATSONX_URL,
    HUGGINGFACEHUB_API_TOKEN
)

from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_ibm import WatsonxLLM
from langchain_community.llms import HuggingFaceHub


# ---------------- LOGGER ----------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def load_llm(mode: str):
    """Return an initialized LLM based on mode selection."""
    if mode == "Watsonx":
        log.info("Initializing IBM Watsonx LLM...")
        model = WatsonxLLM(
            model_id="meta-llama/llama-3-3-70b-instruct",
            url=WATSONX_URL,
            project_id=WATSONX_PROJECT_ID,
            params={"max_new_tokens": 256, "temperature": 0.1},
            apikey=WATSONX_API_KEY
        )
    else:
        log.info("Initializing Hugging Face Falcon model...")
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
        model = HuggingFaceHub(
            repo_id="tiiuae/falcon-7b-instruct",
            model_kwargs={
                "temperature": 0.1,
                "max_new_tokens": 600,
                "max_length": 600
            }
        )
    return model


def prepare_pdf(pdf_file):
    """Load, split, and embed a PDF document."""
    log.info(f"Loading PDF: {pdf_file.name}")

    loader = PyPDFLoader(pdf_file.name)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    chunks = splitter.split_documents(docs)

    log.info(f"Split PDF into {len(chunks)} chunks for embedding.")

    # Create embeddings
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": DEVICE}
    )

    # Create Chroma vector store
    vectordb = Chroma.from_documents(chunks, embedding=embeddings)
    log.info("Chroma vector database created successfully.")
    return vectordb


def generate_response(mode, vectordb, question, chat_history):
    """Use the selected model to answer a question from the indexed PDF."""
    if vectordb is None:
        return chat_history + [[question, "Please upload a PDF and choose a model first."]]

    llm = load_llm(mode)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 6, "lambda_mult": 0.25}),
        return_source_documents=False
    )

    result = qa.invoke({"question": question})
    answer = result["result"]

    chat_history.append([question, answer])
    return chat_history


# =========================================================
# GRADIO INTERFACE
# =========================================================
def main():
    with gr.Blocks(theme=gr.themes.Soft(), title="PDF Question Assistant") as demo:
        gr.Markdown("# 📚 PDF Question Assistant")
        gr.Markdown(
            "Upload a PDF, select a model (Watsonx or Hugging Face), and ask your questions!"
        )

        pdf_file = gr.File(label="Upload your PDF", file_types=[".pdf"])
        question = gr.Textbox(label="Ask a question about your document")
        chatbot = gr.Chatbot(label="Chat History")
        clear_btn = gr.Button("🧹 Clear Chat")

        # Hidden state to store the processed vector DB
        vectordb_state = gr.State()

        # ---------------- Action Buttons ----------------
        with gr.Row():
            watson_btn = gr.Button("🔷 Use Watsonx", variant="primary")
            hf_btn = gr.Button("🤗 Use Hugging Face", variant="secondary")

        # ========== PDF upload handling ==========
        def handle_pdf_upload(pdf):
            if pdf is None:
                return None, "Please upload a valid PDF file."
            vectordb = prepare_pdf(pdf)
            return vectordb, "✅ PDF successfully processed. You can now ask questions!"

        upload_status = gr.Textbox(label="Status", interactive=False)
        pdf_file.upload(fn=handle_pdf_upload, inputs=pdf_file, outputs=[vectordb_state, upload_status])

        # ========== Watsonx button click ==========
        watson_btn.click(
            fn=lambda vdb, q, h: generate_response("Watsonx", vdb, q, h),
            inputs=[vectordb_state, question, chatbot],
            outputs=chatbot
        )

        # ========== Hugging Face button click ==========
        hf_btn.click(
            fn=lambda vdb, q, h: generate_response("Hugging Face", vdb, q, h),
            inputs=[vectordb_state, question, chatbot],
            outputs=chatbot
        )

        # Clear chat
        clear_btn.click(lambda: [], None, chatbot)

    return demo


if __name__ == "__main__":
    app = main()
    app.launch(server_name="0.0.0.0", server_port=7860)
