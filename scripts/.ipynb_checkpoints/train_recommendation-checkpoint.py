import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

def train_model():
    print("Loading processed data...")
    input_path = os.path.join('data', 'processed', 'cleaned_books.csv')
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Please run preprocess.py first.")
        return
        
    books = pd.read_csv(input_path, low_memory=False)
    
    print("Building TF-IDF Vectorizer...")
    # We use TF-IDF on the combined features
    tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
    
    # Fill any NaNs that might have been created
    features = books['Combined-Features'].fillna('')
    tfidf_matrix = tfidf.fit_transform(features)
    
    print("Training NearestNeighbors model...")
    # Using cosine similarity via NearestNeighbors
    model = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
    model.fit(tfidf_matrix)
    
    print("Saving models...")
    os.makedirs('models', exist_ok=True)
    
    # We save both the TF-IDF vectorizer and the NN model
    model_path = os.path.join('models', 'model.pkl')
    vectorizer_path = os.path.join('models', 'vectorizer.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(tfidf, f)
        
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")

if __name__ == "__main__":
    train_model()
