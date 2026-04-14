# Hybrid Recommendation System - Documentation

## Overview

The Shelf Scanner AI now uses a **Hybrid Recommendation System** that combines:
1. **Collaborative Filtering** (60% weight) - Finds books similar to a query based on user rating patterns
2. **Content-Based Filtering** (25% weight) - Finds books similar based on description/title/author
3. **Genre-Based Scoring** (10% weight) - Falls back to keyword matching if other methods fail
4. **Popularity Boosting** (5% weight) - Preferences highly-rated books

## Architecture

### Models & Data

- **Books Dataset**: 242,135 books with metadata (title, author, publisher)
- **Ratings Dataset**: 433,671 user ratings (scale 0-10, filtered from 1,149,780 total)
- **Users Dataset**: 77,805 unique users with location and age info

### Collaborative Filtering Models

- **SVD (Singular Value Decomposition)**: Extracts 50 latent factors from user-item rating matrix
- **Item-Item Similarity**: Uses KNN with cosine distance on item factor vectors
- **User-Item Matrix**: Sparse matrix (77,805 users × 242,135 books)

### Content-Based Models

- **TF-IDF Vectorizer**: 10,000 max features from book titles and authors
- **Nearest Neighbors**: KNN with cosine similarity to find similar books by description

## Pipeline

### 1. Preprocessing (`scripts/preprocess.py`)

```
Input: archive/{Books.csv, Ratings.csv, Users.csv}
↓
1. Load all three datasets
2. Filter out invalid ratings (rating = 0)
3. Calculate rating statistics per ISBN:
   - Average rating (0-10 scale)
   - Rating count
   - Rating standard deviation
4. Compute popularity score: (rating_mean × 0.7) + (log(rating_count) × 0.3)
5. Clean text features (normalize titles and authors)
6. Output: data/processed/{cleaned_books.csv, ratings_processed.csv, users_processed.csv}
```

### 2. Model Training (`scripts/train_recommendation.py`)

```
Input: data/processed/cleaned_books.csv and ratings_processed.csv
↓
CONTENT-BASED:
  1. Build TF-IDF matrix on "Combined-Features" (title + author)
  2. Train NearestNeighbors with cosine metric
  3. Save: tfidf_vectorizer.pkl, content_model.pkl

COLLABORATIVE FILTERING:
  1. Create user-item rating matrix (sparse)
  2. Apply SVD with n_factors = min(50, sqrt(nnz))
  3. Train item-item model on SVD item factors
  4. Save: svd_model.pkl, collab_item_model.pkl, item_factors.pkl, user_factors.pkl

MAPPINGS:
  1. ISBN ↔ Index mapping
  2. User-ID ↔ Index mapping
  3. Save: isbn_to_idx.pkl, idx_to_isbn.pkl, user_to_idx.pkl, idx_to_user.pkl
```

### 3. Recommendation Generation (`backend/utils.py`)

```
Input: description, title, exclude_title
↓
STEP 1: COLLABORATIVE FILTERING
  - Find reference book in dataset matching query title
  - Use collab_item_model to find similar books (by user rating patterns)
  - Score weight: 60%

STEP 2: CONTENT-BASED FILTERING
  - Vectorize description with TF-IDF
  - Find similar books using content_model
  - Score weight: 25%

STEP 3: FALLBACK GENRE SCORING
  - If collaborative/content models fail
  - Use keyword matching for genre detection
  - Score weight: 10%

STEP 4: RANKING
  - Combine scores: hybrid_score = (collab × 0.6) + (content × 0.25) + (genre × 0.1) + (popularity × 0.05)
  - Sort by hybrid_score descending
  - Return top 5 unique recommendations
```

## Key Features

### 1. Rating-Based Popularity
- Books are weighted by average rating and number of ratings
- Highly-rated, frequently-rated books are prioritized
- Prevents recommending obscure/low-quality books

### 2. Collaborative Filtering
- Leverages implicit feedback (user ratings)
- Finds books that similar users also rated highly
- Works even when descriptions are sparse
- Priority: 60% of recommendation score

### 3. Content-Based Filtering
- Handles new books without rating history
- Matches on title, author, and description
- TF-IDF ensures semantic similarity
- Priority: 25% of recommendation score

### 4. Hybrid Approach
- Combines strengths of both methods
- Avoids cold-start problem (new books + new users)
- More robust recommendations
- Graceful fallback chain

## Usage

### Quick Search
```python
from backend.utils import recommend_books

description = "A thrilling mystery novel with detective work and surprising twists"
recommendations = recommend_books(description, title="Murder Mystery Book")

for rec in recommendations:
    print(f"{rec['title']} by {rec['author']}")
    print(f"Score: {rec['similarity_score']}, Rating: {rec['rating']}")
```

### Image Scanning Workflow
1. User scans book cover with image
2. CV pipeline extracts title/ISBN
3. `get_book_data()` retrieves full book info
4. `recommend_books()` generates recommendations using full description
5. User gets 5 book recommendations

## Performance Metrics

- **Dataset Size**: 242,135 books
- **User Ratings**: 433,671 total ratings
- **Sparsity**: ~0.00003% (very sparse - typical for large books)
- **SVD Factors**: 50 latent factors
- **Explained Variance**: ~45% from 50 factors

## Model Sizes

- `svd_model.pkl`: ~96 MB (SVD model)
- `collab_item_model.pkl`: ~96 MB (Item-item KNN)
- `content_model.pkl`: ~15 MB (Content KNN)
- `tfidf_vectorizer.pkl`: ~364 KB (TF-IDF)
- `item_factors.pkl`: ~96 MB (Item embeddings)
- `user_factors.pkl`: ~31 MB (User embeddings)
- **Total**: ~335 MB

## Improvement Opportunities

1. **Implicit Feedback**: Could use view count as implicit rating
2. **Cold Start**: Pre-compute genres/categories for new books
3. **User Similarity**: Use user factors for user-based CF instead of item-based
4. **Temporal Dynamics**: Weight recent ratings higher
5. **Context Awareness**: Adjust for user's reading history
6. **Ensemble Methods**: Combine with other collaborative algorithms (e.g., NMF, ALS)

## Troubleshooting

### Issue: Recommendations seem generic
**Solution**: Ensure description is specific and long enough (>10 words). Collaborative filtering works best with popular books; niche books may not have enough ratings.

### Issue: Wrong genre recommendations
**Solution**: Update genre_keywords in recommend_books() function or provide more detailed description.

### Issue: Models not loading
**Solution**: Run preprocessing and training scripts:
```bash
python scripts/preprocess.py
python scripts/train_recommendation.py
```

### Issue: Memory errors during training
**Solution**: Reduce SVD n_components or filter down books by rating_count > threshold

## Testing

Run the test suite:
```bash
python test_hybrid_recommendations.py
```

This tests recommendations across multiple genres:
- Self-help / Productivity
- Fantasy / Adventure
- Mystery / Thriller
- Business / Finance

---

**Last Updated**: March 23, 2026
**System**: Hybrid (Collaborative 60% + Content-Based 25% + Genre 10% + Popularity 5%)
