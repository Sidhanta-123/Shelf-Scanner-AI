# Shelf Scanner AI

A full-stack AI application that uses computer vision and machine learning to scan a shelf of books using a camera or uploaded image, identifies the books using Tesseract OCR and YOLO, and recommends similar items using an NLP TF-IDF and K-Nearest Neighbors matching pipeline.

## Features

- **Live Camera Scanning**: Uses device camera to capture images of physical books.
- **Image Upload**: Upload images of a bookshelf or single book.
- **YOLO Object Detection**: Detects books within images.
- **Optical Character Recognition**: Extracts title strings from book covers.
- **Fuzzy Search & Machine Learning Matching**: Matches raw OCR output to a book database.
- **Recommendation Engine**: Suggests related books using Cosine Similarity on Title and Author embeddings.
- **Modern UI**: Clean and responsive HTML/JS interface interacting via RESTful APIs.

## Project Structure

- `backend/`: FastAPI Python backend for inference and API endpoints. 
- `frontend/`: Vanilla HTML/JS minimal frontend for UI.
- `scripts/`: Python scripts for data cleaning and model training.
- `archive/`: Raw CSV datasets (Books, Users, Ratings).
- `data/`: Processed outputs used by the models.
- `models/`: Saved `model.pkl` and `vectorizer.pkl`.

## Setup & Running Locally

### Prerequisites
- Python 3.10+
- Tesseract-OCR installed on your system (e.g. `sudo apt install tesseract-ocr` or installing via Windows executable).
- Node.js (Optional, not needed for current simple frontend)

### 1. Data Preparation and Training
Before running the backend, you must preprocess the data and train the recommendation model.

```bash
python scripts/preprocess.py
python scripts/train_recommendation.py
```
This will parse the `archive/*.csv` files, create `data/processed/cleaned_books.csv`, and save the Scikit-Learn models to `models/`.

### 2. Run Backend API (Windows)

Open a new **PowerShell** window and run:
```powershell
cd backend
# Install dependencies (only needed once)
python -m pip install -r requirements.txt

# Start the server (it stays running in a new window)
Start-Process python -ArgumentList "-m uvicorn main:app --host 0.0.0.0 --port 8000" -NoNewWindow
```
The API will be available at `http://localhost:8000`.
You can verify it's running: `Invoke-WebRequest -Uri http://localhost:8000/ -UseBasicParsing`

### 2. Run Backend API (Linux/Mac)
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3. Run Frontend

Open a **second** terminal window:
```bash
cd frontend
python -m http.server 3000
```
Visit `http://localhost:3000` in your browser.

## Docker Deployment
To run the entire stack using Docker Compose:

```bash
docker-compose up --build
```
This spins up the backend on port 8000 and the frontend on port 3000.
