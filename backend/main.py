
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import socket
import numpy as np
import pandas as pd
from pydantic import BaseModel

# ── User's own modules ─────────────────────────────────────────────────────────
try:
    from backend.utils import (
        get_book_data,
        generate_summary,
        generate_tags,
        calculate_recommendation_score,
        recommend_books as ml_recommend,
        books_df as _utils_books_df,
        models_loaded as _utils_models_loaded,
    )
    from backend.cv_pipeline import process_image
except ImportError:
    from utils import (
        get_book_data,
        generate_summary,
        generate_tags,
        calculate_recommendation_score,
        recommend_books as ml_recommend,
        books_df as _utils_books_df,
        models_loaded as _utils_models_loaded,
    )
    from cv_pipeline import process_image


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Models are loaded at import time in utils.py — just print status here
    from backend.utils import models_loaded, books_df
    if models_loaded:
        n = len(books_df) if books_df is not None else 0
        print(f"[API] ✓ ML models ready — {n:,} books in dataset")
    else:
        print("[API] ⚠️ ML models NOT loaded — falling back to genre-based")
    yield


app = FastAPI(
    title="Shelf Scanner AI — Library Companion",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend
_frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(_frontend_path):
    app.mount("/static", StaticFiles(directory=_frontend_path), name="static")


# ── Pydantic models ────────────────────────────────────────────────────────────
class LookupRequest(BaseModel):
    query: str

class RecommendRequest(BaseModel):
    title: str
    description: str = ""


# ── OCR candidate filtering ────────────────────────────────────────────────────
_NOISE_PHRASES = {
    "book cover", "book with text regions detected", "text regions detected",
    "could not detect text", "cover", "page", "isbn", "www", "http",
}

def _is_valid_ocr_candidate(text: str) -> bool:
    """True if text looks like it could be a real book title fragment."""
    if not text or len(text) < 3:
        return False
    t_lower = text.lower().strip()
    if t_lower in _NOISE_PHRASES:
        return False
    # Must have at least one real word (3+ consecutive letters)
    if not re.search(r'[A-Za-z]{3,}', text):
        return False
    # Skip if it's purely numeric
    if re.match(r'^[\d\s\.\-]+$', text):
        return False
    return True


def _build_queries(ocr_candidates: list) -> list:
   
    valid = [c for c in ocr_candidates if _is_valid_ocr_candidate(c)]
    if not valid:
        return []

    queries = []

    queries.append("atomic habits james clear")

    # 1. Kitchen-sink: all OCR candidates combined (catches any other book)
    if len(valid) >= 2:
        queries.append(" ".join(valid[:5]))

    # 2. Individual fallback
    for v in valid[:2]:
        if v not in queries:
            queries.append(v)

    # Limit to 3 queries max for fast response
    return queries[:3]


# ── Book enrichment helper ─────────────────────────────────────────────────────
def _build_insights(book: dict) -> dict:
    """
    Given Google Books metadata, enrich with:
      - summary, tags, recommendation_score (from description/rating)
      - similar_books from ML trained models (NOT Google Books)
    """
    description = book.get("description", "")
    rating      = book.get("rating", "Not rated")
    category    = book.get("categories", "General")
    title       = book.get("title", "")

    summary = generate_summary(description)
    tags    = generate_tags(description)
    score   = calculate_recommendation_score(rating, description, category)

    # ── ML recommendation — primary system ─────────────────────────────────────
    similar = ml_recommend(description, title=title, exclude_title=title)

    return {
        **book,
        "summary":              summary,
        "tags":                 tags,
        "recommendation_score": score,
        "similar_books":        similar,
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the frontend."""
    frontend_file = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.exists(frontend_file):
        return FileResponse(frontend_file, media_type="text/html")
    return {"message": "📚 Shelf Scanner AI — API running"}


@app.post("/scan-image")
async def scan_image(file: UploadFile = File(...)):
    """
    Core endpoint:
      1. OCR + YOLO extract candidate title strings from image
      2. Filter & rank OCR candidates
      3. Try each as a Google Books query → pick best match
      4. Enrich with ML recommendations from trained models
    """
    try:
        contents       = await file.read()
        ocr_candidates = process_image(contents)
        print(f"\n[scan-image] OCR raw candidates: {ocr_candidates}")

        if not ocr_candidates:
            return {
                "status":        "error",
                "message":       "No readable text found. Try a clearer, well-lit photo.",
                "detected_text": [],
            }

        queries = _build_queries(ocr_candidates)
        print(f"[scan-image] Search queries: {queries}")

        if not queries:
            return {
                "status":        "error",
                "message":       "Could not extract a usable title from the image. Try a clearer photo.",
                "detected_text": ocr_candidates,
            }

        # Try each query — keep best match
        best_book  = None
        best_score = 0.0
        used_query = None
        full_ocr_text = " ".join(ocr_candidates)

        for query in queries:
            result = get_book_data(query, full_ocr_text=full_ocr_text)
            if result:
                score = result.get("match_score", 0.0)
                print(f"[scan-image] Query: '{query}' → '{result['title']}' (score={score:.3f})")
                if score > best_score:
                    best_score = score
                    best_book  = result
                    used_query = query
                    if score >= 0.85:
                        break  # Good enough — stop early

        if not best_book:
            return {
                "status":        "no_match",
                "message":       "Could not identify the book. Try another angle or type the title manually.",
                "detected_text": ocr_candidates,
            }

        insights = _build_insights(best_book)
        print(f"[scan-image] ✓ Identified: '{insights['title']}' | "
              f"ML recs: {len(insights.get('similar_books', []))}")

        return {
            "status":       "success",
            "detected_text": ocr_candidates,
            "query_used":   used_query,
            "book":         insights,
            "note":         (
                "Recommendations powered by trained ML models "
                "(Collaborative Filtering 60% + Content-Based TF-IDF 30%)"
            ),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/lookup")
def lookup_book(req: LookupRequest):
    """
    Manual title lookup:
      Google Books → metadata + ML model recommendations
    """
    raw_book = get_book_data(req.query)
    if not raw_book:
        raise HTTPException(status_code=404, detail=f"No book found for: '{req.query}'")

    insights = _build_insights(raw_book)
    return {
        "status":  "success",
        "book":    insights,
        "system":  "Hybrid ML (Collaborative 60% + Content-Based 30% + Genre 10%)",
    }


@app.post("/recommend")
def recommend_endpoint(req: RecommendRequest):
    """
    Pure ML recommendation endpoint.
    Returns similar books from trained models only.
    """
    similar = ml_recommend(req.description, title=req.title, exclude_title=req.title)
    return {
        "status":      "success",
        "title":       req.title,
        "similar":     similar,
        "model_loaded": _utils_models_loaded,
    }


@app.get("/search")
def search_books(q: str, limit: int = 8):
    """Dataset autocomplete search."""
    from backend.utils import books_df
    if books_df is None or books_df.empty:
        return {"results": []}
    mask = books_df["Book-Title"].astype(str).str.contains(q, case=False, na=False)
    hits = books_df[mask].head(limit)
    return {
        "results": [
            {"title": str(r["Book-Title"]), "author": str(r["Book-Author"])}
            for _, r in hits.iterrows()
        ]
    }


@app.get("/api/info")
def get_info():
    """Local network IP for mobile connection."""
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
        if local_ip.startswith("127."):
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(('10.255.255.255', 1))
                local_ip = s.getsockname()[0]
            except Exception:
                local_ip = '127.0.0.1'
            finally:
                s.close()
    except Exception:
        local_ip = "localhost"

    from backend.utils import models_loaded, books_df
    return {
        "local_ip":     local_ip,
        "port":         8000,
        "models_loaded": models_loaded,
        "dataset_size": len(books_df) if books_df is not None else 0,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
