import logging
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import watson_worker  
from config import (
    WATSONX_API_KEY,
    WATSONX_PROJECT_ID,
    WATSONX_URL
)

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
        log.info(f"Received message from user: {user_message}")

        bot_reply = watson_worker.handle_query(user_message)
        log.info("Generated response successfully.")

        return jsonify({"botResponse": bot_reply}), 200

    except Exception as e:
        log.exception("Error while processing user message.")
        return jsonify({"botResponse": f"An error occurred: {str(e)}"}), 500


@app.route('/process-document', methods=['POST'])
def process_document_route():
    """Handle PDF uploads and start document processing."""
    if 'file' not in request.files:
        log.warning("No file found in upload request.")
        return jsonify({
            "botResponse": (
                "It seems like the file was not uploaded correctly. Please try again. "
                "If the issue persists, use a different file."
            )
        }), 400

    try:
        uploaded_file = request.files['file']
        file_path = uploaded_file.filename
        uploaded_file.save(file_path)
        log.info(f"File saved locally as: {file_path}")

        watson_worker.ingest_pdf(file_path)
        log.info("PDF processed and indexed successfully.")

        return jsonify({
            "botResponse": (
                "Your PDF has been processed successfully. "
                "You can now ask me any questions related to its content!"
            )
        }), 200

    except Exception as e:
        log.exception("Error while processing document upload.")
        return jsonify({"botResponse": f"Error while processing file: {str(e)}"}), 500


# ---------------------- APP STARTUP ----------------------
if __name__ == "__main__":
    # Initialize backend model and embeddings
    watson_worker.initialize_model()
    log.info("Watson Worker initialized successfully. Server is now running...")

    app.run(debug=True, port=8000, host='0.0.0.0')
