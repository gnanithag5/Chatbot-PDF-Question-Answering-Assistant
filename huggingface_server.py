import logging
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import huggingface_worker  
from config import HUGGINGFACEHUB_API_TOKEN

# ---------------------- FLASK SETUP ----------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
app.logger.setLevel(logging.INFO)

# ---------------------- ROUTES ----------------------

@app.route('/', methods=['GET'])
def index():
    """Serve the frontend HTML interface."""
    log.info("Rendering index.html")
    return render_template('index.html')


@app.route('/process-message', methods=['POST'])
def process_message_route():
    """Handle chat messages sent from the frontend."""
    try:
        user_message = request.json.get('userMessage', '')
        log.info(f"Received user message: {user_message}")

        bot_reply = huggingface_worker.handle_query(user_message)
        log.info("Generated Hugging Face model response successfully.")

        return jsonify({"botResponse": bot_reply}), 200

    except Exception as e:
        log.exception("Error while processing message in Hugging Face backend.")
        return jsonify({"botResponse": f"An error occurred while processing your message: {str(e)}"}), 500


@app.route('/process-document', methods=['POST'])
def process_document_route():
    """Handle uploaded PDF files and process them for retrieval."""
    if 'file' not in request.files:
        log.warning("No file detected in upload request.")
        return jsonify({
            "botResponse": (
                "It seems like the file was not uploaded correctly. Please try again. "
                "If the issue continues, try a different file."
            )
        }), 400

    try:
        uploaded_file = request.files['file']
        file_path = uploaded_file.filename
        uploaded_file.save(file_path)
        log.info(f"Uploaded file saved locally as: {file_path}")

        huggingface_worker.ingest_pdf(file_path)
        log.info("PDF successfully processed and embedded using Hugging Face embeddings.")

        return jsonify({
            "botResponse": (
                "Your document has been analyzed successfully using the Hugging Face model. "
                "You can now ask questions related to its contents!"
            )
        }), 200

    except Exception as e:
        log.exception("Error occurred during document processing in Hugging Face backend.")
        return jsonify({"botResponse": f"Error while processing file: {str(e)}"}), 500


# ---------------------- APP STARTUP ----------------------
if __name__ == "__main__":
    # Initialize Hugging Face model and embeddings
    huggingface_worker.initialize_model()
    log.info("Hugging Face Worker initialized successfully. Server is now running...")

    app.run(debug=True, port=8000, host='0.0.0.0')
