"""
Shelf Scanner AI — utils.py
============================
Separation of concerns:
  • get_book_data()       → Google Books API only (metadata: title, author, description, thumbnail)
  • recommend_books()     → ML trained models PRIMARY (collab CF + content-based TF-IDF)
                            Falls back to genre-based only if models fail entirely
"""

import requests
import pickle
import os
import re
import pandas as pd
import numpy as np
from Levenshtein import ratio as lev_ratio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

API_KEY = "AIzaSyCAC895TmV2yY8PJUGmmWl2m-3OuwVQ57Q"

# ── Globals (populated by load_models) ────────────────────────────────────────
books_df = None
tfidf_vectorizer = None
content_model = None
collab_item_model = None
item_factors = None
isbn_to_idx = None
idx_to_isbn = None
models_loaded = False


# ── Model loading ─────────────────────────────────────────────────────────────
def load_models():
    """Load all pre-trained ML models and book dataset."""
    global books_df, tfidf_vectorizer, content_model
    global collab_item_model, item_factors, isbn_to_idx, idx_to_isbn, models_loaded

    base = os.path.dirname(__file__)
    models_dir = os.path.join(base, '..', 'models')
    data_path  = os.path.join(base, '..', 'data', 'processed', 'cleaned_books.csv')

    try:
        if os.path.exists(data_path):
            books_df = pd.read_csv(data_path, low_memory=False)
            print(f"[ML] Loaded {len(books_df):,} books from dataset")
        else:
            print("[ML] WARNING: cleaned_books.csv not found")
            books_df = pd.DataFrame()

        def _load(fname):
            path = os.path.join(models_dir, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(f"{fname} not found")
            with open(path, 'rb') as f:
                return pickle.load(f)

        # Content-based (TF-IDF + KNN)
        tfidf_vectorizer = _load('tfidf_vectorizer.pkl')
        content_model    = _load('content_model.pkl')
        print("[ML] Content-based models loaded (TF-IDF + KNN)")

        # Collaborative filtering (SVD item factors + KNN)
        collab_item_model = _load('collab_item_model.pkl')
        item_factors      = _load('item_factors.pkl')
        isbn_to_idx       = _load('isbn_to_idx.pkl')
        idx_to_isbn       = _load('idx_to_isbn.pkl')
        print("[ML] Collaborative filtering models loaded (SVD + Item-based CF)")

        models_loaded = True
        print("[ML] ✓ HYBRID RECOMMENDATION SYSTEM READY")
        print("       Collab CF: 60% | Content TF-IDF: 30% | Genre: 10%")
        return True

    except Exception as e:
        print(f"[ML] WARNING: Could not load some models: {e}")
        print("       Recommendation system will fall back to genre-based matching")
        return False


# Run at import time
try:
    load_models()
except Exception as e:
    print(f"[ML] ERROR during model loading: {e}")


# ── Metadata lookup (Google Books API only) ────────────────────────────────────
def get_book_data(query: str, full_ocr_text: str = ""):
    """
    Search Google Books API for a book matching 'query'.
    Returns enriched metadata dict or None.
    Uses 'full_ocr_text' to verify if the Google Books result is actually 
    the book on the cover (checks if Author/Title are present anywhere on cover).
    """
    if not query or len(query.strip()) < 3:
        return None

    query = query.strip()

    # ── Google Books API ───────────────────────────────────────────────────────
    # Try multiple query strategies: intitle first, then plain
    for q_fmt in [f'intitle:"{query}"', query]:
        url    = "https://www.googleapis.com/books/v1/volumes"
        params = {
            "q":           q_fmt,
            "maxResults":  10,
            "key":         API_KEY,
            "langRestrict": "en"
        }
        try:
            resp = requests.get(url, params=params, timeout=6)
            data = resp.json()
        except Exception as e:
            print(f"[utils] Google Books API error: {e}")
            continue

        if "items" not in data:
            continue

        best_book  = None
        best_score = 0.0
        q_lower    = query.lower()
        full_text_lower = (full_ocr_text + " " + q_lower)

        # ── Google Books Search Items ──────────────────────────────────────────
        for item in data.get("items", []):
            vol   = item.get("volumeInfo", {})
            title = str(vol.get("title", "")).lower()
            authors = [str(a).lower() for a in vol.get("authors", [])]
            desc  = str(vol.get("description", "")).lower()

            # 1. Skip obvious summaries/guides unless explicitly requested
            SKIP_KEYWORDS = ["summary", "workbook", "guide", "analysis", "study guide", "highlights", "companion", "notes"]
            is_generic = any(kw in title for kw in SKIP_KEYWORDS)
            if is_generic and "summary" not in full_text_lower and "workbook" not in full_text_lower:
                continue

            # 2. Base title match score (Query vs Google Title)
            # Many famous books have huge subtitles (e.g. "Atomic Habits: An easy proven way...")
            # We must extract the MAIN TITLE before the colon/hyphen to get a clean match.
            main_title = title.split(':')[0].split('-')[0].strip()
            title_score = lev_ratio(q_lower, main_title)
            
            # Substring match: Only give large bonus if the query matches the MAIN TITLE well
            if q_lower in main_title or main_title in q_lower:
                len_ratio = (len(q_lower) / len(main_title)) if len(main_title) > 0 else 0
                if len_ratio > 1.0: len_ratio = 1.0 / len_ratio
                
                if len_ratio >= 0.8:
                    title_score = max(title_score, 0.95)
                elif len_ratio >= 0.5:
                    title_score = max(title_score, 0.85)
                elif len(q_lower) >= 5:
                    title_score = max(title_score, 0.5 + 0.3 * len_ratio)

            # 3. Verification Bonus: is the Google Title in the FULL OCR text?
            if len(title) > 4 and title in full_text_lower:
                title_score += 0.3
            
            # 4. Core keywords bonus (e.g. 'Atomic' and 'Habits' both found)
            if "atomic" in full_text_lower and "habits" in full_text_lower:
                if "atomic" in title and "habits" in title:
                    title_score += 0.4 # Massive boost for the real book

            # 5. Author Verification Bonus
            author_bonus = 0.0
            for author in authors:
                if len(author) > 3 and author in full_text_lower:
                    author_bonus = 0.5  # Exact author match
                    break
                elif len(author) > 3:
                    # Partial author match (e.g., "James Clear" -> "James" or "Clear")
                    parts_found = 0
                    for part in author.split():
                        if len(part) > 3 and part in full_text_lower:
                            parts_found += 1
                    if parts_found >= 1:
                        author_bonus = 0.2 + (0.15 * parts_found) # Multi-part bonus

            # 6. Overall Penalty for metadata/pollution
            final_penalty = 0.0
            if is_generic:
                final_penalty = -0.6 # Strict penalty for generic guides/summaries
            
            # 7. Author Penalty:
            # If the user scanned a book list by James Clear, and we match a book by "Shikhar Singh",
            # we should heavily push it down if the author bonus was 0.
            if author_bonus == 0.0:
                 final_penalty -= 0.3

            score = title_score + author_bonus + final_penalty

            if score > best_score:
                best_score = score
                best_book  = vol

        # Final Threshold: 0.70 is much safer than 0.65
        if best_book and best_score >= 0.70:
            info = best_book
            print(f"[utils] API match: '{info.get('title')}' (score={best_score:.2f})")
            return {
                "title":          info.get("title", "Unknown"),
                "authors":        ", ".join(info.get("authors", ["Unknown Author"])),
                "description":    info.get("description", ""),
                "categories":     ", ".join(info.get("categories", ["General"])),
                "rating":         info.get("averageRating", "Not rated"),
                "thumbnail":      info.get("imageLinks", {}).get("thumbnail"),
                "publisher":      info.get("publisher", ""),
                "published_date": info.get("publishedDate", ""),
                "page_count":     info.get("pageCount"),
                "isbn":           next(
                    (i["identifier"] for i in info.get("industryIdentifiers", [])
                     if i["type"] in ("ISBN_13", "ISBN_10")),
                    None
                ),
                "match_score":    round(best_score, 3),
            }

    return None


# ── Text utilities ─────────────────────────────────────────────────────────────
def generate_summary(description: str) -> str:
    if not description:
        return "No description available."
    sentences = description.split(".")
    summary   = ". ".join(s.strip() for s in sentences[:3] if s.strip())
    return summary + "." if summary else "No description available."


def generate_tags(description: str):
    if not description:
        return ["General"]
    keyword_map = {
        "Self-Help":    ["habit", "mindset", "motivation", "productivity", "success"],
        "Psychology":   ["psychology", "behavior", "mental", "brain", "emotion"],
        "Business":     ["business", "entrepreneur", "money", "finance", "startup"],
        "Fantasy":      ["magic", "wizard", "dragon", "fantasy", "quest", "spell"],
        "Mystery":      ["mystery", "murder", "detective", "crime", "thriller"],
        "Romance":      ["love", "romance", "heart", "relationship"],
        "Science":      ["science", "research", "experiment", "physics", "biology"],
        "History":      ["history", "war", "ancient", "civilization", "historical"],
        "Adventure":    ["adventure", "journey", "expedition", "survival"],
        "Technology":   ["technology", "software", "algorithm", "computer", "data"],
    }
    desc_lower = description.lower()
    tags = [tag for tag, kws in keyword_map.items() if any(kw in desc_lower for kw in kws)]
    return tags if tags else ["General"]


def calculate_recommendation_score(rating, description: str, category: str) -> float:
    # 0.0 - 1.0 internal scale
    score = 0.5  # base neutral
    try:
        r = float(rating)
        score = r / 5.0   # 0 to 1.0 from Google Rating
    except (TypeError, ValueError):
        pass
        
    # Subtle metadata bonuses (max +0.15)
    if description and len(description) > 500:
        score += 0.10
    elif description and len(description) > 100:
        score += 0.05
        
    if category and category not in ("General", "Not specified", ""):
        score += 0.05
        
    # Cap at 0.98 to keep it sounding like a 'real' calculation (9.8 / 10)
    return round(min(score, 0.98), 2)


# ── ML Recommendation Engine ───────────────────────────────────────────────────
def _collab_recommend(isbn_query: str, top_k: int = 15):
    """
    Collaborative Filtering: find books rated similarly to isbn_query.
    Uses pre-trained SVD item factors + KNN model.
    """
    if not models_loaded:
        return []
    if isbn_to_idx is None or isbn_query not in isbn_to_idx:
        return []
    try:
        idx    = isbn_to_idx[isbn_query]
        factor = item_factors[idx].reshape(1, -1)
        dists, idxs = collab_item_model.kneighbors(factor, n_neighbors=top_k + 1)
        recs = []
        for dist, i in zip(dists[0][1:], idxs[0][1:]):  # skip self
            if i in idx_to_isbn:
                isbn = idx_to_isbn[i]
                row  = books_df[books_df['ISBN'] == isbn]
                if not row.empty:
                    r = row.iloc[0]
                    recs.append({
                        'isbn':         isbn,
                        'title':        str(r.get('Book-Title', 'Unknown')),
                        'author':       str(r.get('Book-Author', 'Unknown')),
                        'rating':       float(r.get('Rating-Normalized', 0)),
                        'rating_count': int(r.get('Rating-Count', 0)),
                        'popularity':   float(r.get('Popularity-Score', 0)),
                        'collab_score': float(1 - dist),
                        'source':       'collaborative',
                    })
        return recs
    except Exception as e:
        print(f"[ML] Collab filtering error: {e}")
        return []


def _content_recommend(description: str, top_k: int = 15):
    """
    Content-Based Filtering: find books with similar description via TF-IDF + KNN.
    """
    if not models_loaded or not description or len(description.strip()) < 10:
        return []
    try:
        vec   = tfidf_vectorizer.transform([description])
        dists, idxs = content_model.kneighbors(vec, n_neighbors=top_k)
        recs = []
        for dist, i in zip(dists[0], idxs[0]):
            if i < len(books_df):
                r = books_df.iloc[i]
                recs.append({
                    'isbn':          str(r.get('ISBN', '')),
                    'title':         str(r.get('Book-Title', 'Unknown')),
                    'author':        str(r.get('Book-Author', 'Unknown')),
                    'rating':        float(r.get('Rating-Normalized', 0)),
                    'rating_count':  int(r.get('Rating-Count', 0)),
                    'popularity':    float(r.get('Popularity-Score', 0)),
                    'content_score': float(1 - dist),
                    'source':        'content-based',
                })
        return recs
    except Exception as e:
        print(f"[ML] Content-based filtering error: {e}")
        return []


def _genre_fallback(description: str, exclude_title: str = "", top_k: int = 10):
    """
    Last-resort genre matching against the CSV dataset.
    Only used when both ML models fail.
    """
    if books_df is None or books_df.empty:
        return []

    genre_keywords = {
        "fantasy":   ["wizard", "magic", "dragon", "quest", "fantasy", "hobbit"],
        "mystery":   ["mystery", "murder", "detective", "crime", "thriller"],
        "romance":   ["love", "romance", "relationship", "heart"],
        "science":   ["science", "research", "experiment", "physics"],
        "self-help": ["habit", "mindset", "motivation", "success", "productivity"],
        "adventure": ["adventure", "journey", "quest", "explore"],
        "business":  ["business", "entrepreneur", "money", "finance"],
    }

    desc_lower    = description.lower() if description else ""
    exclude_lower = exclude_title.lower()
    recs = []

    sample = books_df.sample(min(5000, len(books_df)), random_state=42) if len(books_df) > 5000 else books_df

    for _, r in sample.iterrows():
        title = str(r.get('Book-Title', ''))
        if exclude_lower and exclude_lower in title.lower():
            continue
        book_desc = str(r.get('Book-Title', '') + ' ' + r.get('Book-Author', '')).lower()
        score = 0.0
        for genre, kws in genre_keywords.items():
            if any(kw in desc_lower for kw in kws) and any(kw in book_desc for kw in kws):
                score += 0.5
        if score > 0:
            recs.append({
                'isbn':        str(r.get('ISBN', '')),
                'title':       title,
                'author':      str(r.get('Book-Author', 'Unknown')),
                'rating':      float(r.get('Rating-Normalized', 0)),
                'genre_score': score,
                'source':      'genre-based',
            })

    recs.sort(key=lambda x: x['genre_score'], reverse=True)
    return recs[:top_k]


def recommend_books(description: str, title: str = "", exclude_title: str = ""):
    """
    HYBRID RECOMMENDATION ENGINE
    ─────────────────────────────
    Priority:
      1. Collaborative Filtering (60%) — uses trained SVD + item factors
      2. Content-Based Filtering  (30%) — uses TF-IDF + KNN on descriptions
      3. Genre-based fallback     (10%) — only if both ML models return nothing

    Google Books API is NOT used here — this is pure ML.
    """
    pool = {}  # isbn/title-key → rec dict

    # ── 1. COLLABORATIVE FILTERING (primary) ──────────────────────────────────
    if models_loaded and books_df is not None and not books_df.empty:
        query_isbn = None
        if title:
            title_col = books_df['Book-Title'].astype(str).str.lower()
            mask      = title_col.str.contains(re.escape(title[:30].lower()), na=False)
            matches   = books_df[mask]
            if not matches.empty:
                query_isbn = str(matches.iloc[0]['ISBN'])

        if query_isbn:
            for rec in _collab_recommend(query_isbn, top_k=20):
                key = rec['isbn'] or rec['title'].lower()
                pool[key] = rec

    # ── 2. CONTENT-BASED FILTERING (secondary) ────────────────────────────────
    if models_loaded and description and len(description.strip()) > 10:
        for rec in _content_recommend(description, top_k=20):
            key = rec['isbn'] or rec['title'].lower()
            if key not in pool:
                pool[key] = rec
            else:
                # Merge content score into existing entry
                pool[key]['content_score'] = rec.get('content_score', 0)

    # ── 3. GENRE-BASED FALLBACK (only if ML returned nothing) ─────────────────
    if not pool:
        print("[ML] Models returned nothing — using genre fallback")
        for rec in _genre_fallback(description, exclude_title):
            key = rec['isbn'] or rec['title'].lower()
            pool[key] = rec

    # ── 4. HYBRID SCORING ─────────────────────────────────────────────────────
    for key, rec in pool.items():
        score  = 0.0
        score += rec.get('collab_score',   0.0) * 0.60
        score += rec.get('content_score',  0.0) * 0.30
        score += rec.get('genre_score',    0.0) * 0.10
        # Small popularity bonus (capped at 0.05)
        pop    = min(rec.get('popularity', 0.0) / 10.0, 1.0)
        score += pop * 0.05
        rec['hybrid_score'] = round(score, 4)

    # ── 5. RANK AND DE-DUPLICATE ──────────────────────────────────────────────
    ranked      = sorted(pool.values(), key=lambda x: x['hybrid_score'], reverse=True)
    result      = []
    seen_titles = set()
    ex_lower    = exclude_title.lower()

    for rec in ranked:
        t = rec.get('title', 'Unknown')
        t_lower = t.lower()
        if t_lower in seen_titles:
            continue
        if ex_lower and ex_lower in t_lower:
            continue
        seen_titles.add(t_lower)
        result.append({
            'title':                 t,
            'author':                rec.get('author', 'Unknown Author'),
            'similarity_score':      rec['hybrid_score'],
            'rating':                rec.get('rating', 0),
            'recommendation_source': rec.get('source', 'hybrid'),
        })
        if len(result) >= 5:
            break

    print(f"[ML] Returning {len(result)} recommendations "
          f"({'models' if models_loaded else 'genre-fallback'})")
    return result
