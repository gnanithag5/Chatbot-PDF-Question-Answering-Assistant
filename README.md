# Chatbot-PDF-Question-Answering-Assistant


A smart web application that lets you **upload a PDF**, and then **ask questions** or get **summaries** from its content.  
You can run it in **two modes**:
- **Watson Mode** – uses IBM Watsonx LLM (requires Watson credentials)  
- **Hugging Face Mode** – uses open-source models from the Hugging Face Hub  

Built with **Flask**, **LangChain**, **Chroma**, and **Python 3.10**.

---

## Features
- Upload any `.pdf` document and extract contextual insights
- Query, summarize, or analyze PDF contents interactively
- Switch between **Watsonx** and **Hugging Face** backends
- Vector-based retrieval using **ChromaDB**
- Frontend built with **HTML, Bootstrap, and JavaScript**

---

## Setup Instructions

**1️. Clone the Repository**
**2. Create Virtual Environment**
**3. Install Dependencies from requirements.txt**

---

## Run Locally (Without Docker)
Option 1: Run Watsonx Mode
python watson_server.py


Access in your browser at http://localhost:8000

Option 2: Run Hugging Face Mode
python huggingface_server.py


Access in your browser at http://localhost:8000

## Run with Docker

The Dockerfile supports two modes using the environment variable APP_MODE.

1️. Build the Docker Image
docker build -t pdf-assistant .

2️. Run in Watson Mode
docker run -p 8000:8000 --env-file .env -e APP_MODE=watson pdf-assistant

3️. Run in Hugging Face Mode
docker run -p 8000:8000 --env-file .env -e APP_MODE=huggingface pdf-assistant

Default behavior

If you don’t specify APP_MODE, it defaults to huggingface.

