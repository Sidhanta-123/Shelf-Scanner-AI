import pandas as pd
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

def train_model():
    print("Loading processed data...")
    input_path = os.path.join('data', 'processed', 'cleaned_books.csv')
    ratings_path = os.path.join('data', 'processed', 'ratings_processed.csv')
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Please run preprocess.py first.")
        return
    
    books = pd.read_csv(input_path, low_memory=False)
    ratings = pd.read_csv(ratings_path, low_memory=False)
    
    print(f"Loaded {len(books)} books and {len(ratings)} ratings")
    
    # ==================== CONTENT-BASED MODELS ====================
    print("\n=== Training Content-Based Models ===")
    print("Building TF-IDF Vectorizer...")
    
    # TF-IDF for similar books (based on title + author)
    tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
    features = books['Combined-Features'].fillna('')
    tfidf_matrix = tfidf.fit_transform(features)
    
    print("Training NearestNeighbors for content-based search...")
    content_model = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
    content_model.fit(tfidf_matrix)
    
    # ==================== COLLABORATIVE FILTERING MODELS ====================
    print("\n=== Training Collaborative Filtering Models ===")
    
    # Create ISBN to index mapping
    isbn_to_idx = {isbn: idx for idx, isbn in enumerate(books['ISBN'].unique())}
    idx_to_isbn = {idx: isbn for isbn, idx in isbn_to_idx.items()}
    
    # Create user to index mapping
    user_ids = ratings['User-ID'].unique()
    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    idx_to_user = {idx: uid for uid, idx in user_to_idx.items()}
    
    print(f"Created mappings: {len(isbn_to_idx)} books, {len(user_to_idx)} users")
    
    # Build user-item rating matrix (sparse matrix for memory efficiency)
    print("Building user-item rating matrix...")
    user_indices = ratings['User-ID'].map(user_to_idx).values
    item_indices = ratings['ISBN'].map(isbn_to_idx).values
    values = ratings['Book-Rating'].values
    
    # Handle NaN values in indices (ISBNs not in our book list)
    valid_mask = ~(pd.isna(user_indices) | pd.isna(item_indices))
    user_indices = user_indices[valid_mask].astype(int)
    item_indices = item_indices[valid_mask].astype(int)
    values = values[valid_mask]
    
    rating_matrix = csr_matrix((values, (user_indices, item_indices)), 
                               shape=(len(user_to_idx), len(isbn_to_idx)))
    print(f"Rating matrix shape: {rating_matrix.shape}, Sparsity: {1 - rating_matrix.nnz / (rating_matrix.shape[0] * rating_matrix.shape[1]):.4f}")
    
    # SVD for collaborative filtering
    print("Training SVD for collaborative filtering...")
    n_factors = min(50, int(np.sqrt(rating_matrix.nnz)))  # Adaptive number of factors
    svd = TruncatedSVD(n_components=n_factors, random_state=42, n_iter=100)
    user_factors = svd.fit_transform(rating_matrix)
    item_factors = svd.components_.T
    
    print(f"SVD factors: {n_factors}, explained variance: {svd.explained_variance_ratio_.sum():.4f}")
    
    # Train NearestNeighbors on SVD item factors for item-based collaborative filtering
    collab_item_model = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
    collab_item_model.fit(item_factors)
    
    # ==================== SAVE ALL MODELS ====================
    print("\n=== Saving Models ===")
    os.makedirs('models', exist_ok=True)
    
    # Content-based
    with open(os.path.join('models', 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(tfidf, f)
    with open(os.path.join('models', 'content_model.pkl'), 'wb') as f:
        pickle.dump(content_model, f)
    
    # Collaborative filtering
    with open(os.path.join('models', 'svd_model.pkl'), 'wb') as f:
        pickle.dump(svd, f)
    with open(os.path.join('models', 'collab_item_model.pkl'), 'wb') as f:
        pickle.dump(collab_item_model, f)
    with open(os.path.join('models', 'user_factors.pkl'), 'wb') as f:
        pickle.dump(user_factors, f)
    with open(os.path.join('models', 'item_factors.pkl'), 'wb') as f:
        pickle.dump(item_factors, f)
    
    # Mappings
    with open(os.path.join('models', 'isbn_to_idx.pkl'), 'wb') as f:
        pickle.dump(isbn_to_idx, f)
    with open(os.path.join('models', 'idx_to_isbn.pkl'), 'wb') as f:
        pickle.dump(idx_to_isbn, f)
    with open(os.path.join('models', 'user_to_idx.pkl'), 'wb') as f:
        pickle.dump(user_to_idx, f)
    with open(os.path.join('models', 'idx_to_user.pkl'), 'wb') as f:
        pickle.dump(idx_to_user, f)
    
    print("✓ All models saved successfully!")
    print(f"  - Content-based: tfidf_vectorizer.pkl, content_model.pkl")
    print(f"  - Collaborative: svd_model.pkl, collab_item_model.pkl, user/item_factors.pkl")
    print(f"  - Mappings: isbn_to_idx.pkl, user_to_idx.pkl, etc.")

if __name__ == "__main__":
    train_model()
