"""Quick verification script for the fixed pipeline."""
import sys
sys.path.insert(0, '.')

print("=" * 60)
print("SHELF SCANNER AI - Pipeline Verification")
print("=" * 60)

# ── 1. Test utils loading ─────────────────────────────────────────────────────
print("\n[1] Testing utils.py model loading...")
try:
    from backend.utils import models_loaded, books_df, recommend_books, get_book_data
    print(f"    models_loaded : {models_loaded}")
    print(f"    books_df size : {len(books_df) if books_df is not None else 0:,}")
except Exception as e:
    print(f"    ERROR: {e}")
    sys.exit(1)

# ── 2. Test Google Books metadata lookup ──────────────────────────────────────
print("\n[2] Testing get_book_data (Google Books metadata)...")
for query in ["Atomic Habits", "Harry Potter", "The Hobbit"]:
    try:
        book = get_book_data(query)
        if book:
            print(f"    '{query}' → '{book['title']}' by {book['authors']} (score={book.get('match_score', '?')})")
        else:
            print(f"    '{query}' → No match found")
    except Exception as e:
        print(f"    '{query}' → ERROR: {e}")

# ── 3. Test ML recommendations ────────────────────────────────────────────────
print("\n[3] Testing ML recommendations (trained models)...")
test_cases = [
    ("Atomic Habits", "habits productivity self improvement mindset success"),
    ("Harry Potter", "wizard magic school fantasy adventure dark forces"),
]
for title, desc in test_cases:
    try:
        recs = recommend_books(desc, title=title, exclude_title=title)
        print(f"    '{title}' → {len(recs)} recs:")
        for r in recs[:3]:
            print(f"        - {r['title']} [{r['recommendation_source']}, score={r['similarity_score']:.3f}]")
    except Exception as e:
        print(f"    '{title}' → ERROR: {e}")

# ── 4. Test cv_pipeline import ────────────────────────────────────────────────
print("\n[4] Testing cv_pipeline.py import...")
try:
    from backend.cv_pipeline import process_image, _has_easyocr, _has_tesseract, _has_yolo
    print(f"    EasyOCR    : {'✓' if _has_easyocr else '✗'}")
    print(f"    Tesseract  : {'✓' if _has_tesseract else '✗'}")
    print(f"    YOLO       : {'✓' if _has_yolo else '✗'}")
    print("    process_image imported OK")
except Exception as e:
    print(f"    ERROR: {e}")

print("\n" + "=" * 60)
print("Verification complete.")
print("=" * 60)
