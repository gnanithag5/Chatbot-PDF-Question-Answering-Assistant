FROM python:3.10

# Set working directory
WORKDIR /app

# Copy all project files
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Environment variable to select backend (default = huggingface)
ENV APP_MODE=huggingface

# Entry command
CMD ["bash", "-c", "python -u ${APP_MODE}_server.py"]
