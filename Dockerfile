FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and Tesseract
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements
COPY backend/requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY backend/ ./backend/

# Copy models and processed data so they are accessible by backend
# (We assume train_recommendation.py was run locally beforehand)
COPY models/ ./models/
COPY data/processed/cleaned_books.csv ./data/processed/

# Set working directory to backend
WORKDIR /app/backend

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
