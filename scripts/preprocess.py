import pandas as pd
import numpy as np
import os
import re

def clean_text(text):
    if pd.isna(text):
        return ""
    # Lowercase text and remove non-alphanumeric characters
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def preprocess_data():
    print("Loading datasets...")
    # Load all three datasets
    books_path = os.path.join('archive', 'Books.csv')
    ratings_path = os.path.join('archive', 'Ratings.csv')
    users_path = os.path.join('archive', 'Users.csv')
    
    books = pd.read_csv(books_path, low_memory=False, on_bad_lines='skip')
    ratings = pd.read_csv(ratings_path, low_memory=False)
    users = pd.read_csv(users_path, low_memory=False)
    
    print(f"Original books count: {len(books)}")
    print(f"Ratings count: {len(ratings)}")
    print(f"Users count: {len(users)}")
    
    # ===== BOOK PROCESSING =====
    # Drop duplicates based on Book-Title
    books = books.drop_duplicates(subset='Book-Title', keep='first')
    
    # Handle missing values
    books['Book-Author'] = books['Book-Author'].fillna('Unknown')
    books['Publisher'] = books['Publisher'].fillna('Unknown')
    
    # Clean text features for content-based filtering
    print("Cleaning text features...")
    books['Clean-Title'] = books['Book-Title'].apply(clean_text)
    books['Clean-Author'] = books['Book-Author'].apply(clean_text)
    books['Combined-Features'] = books['Clean-Title'] + " " + books['Clean-Author']
    
    # ===== RATINGS PROCESSING =====
    # Filter out ratings of 0 (which indicate no actual rating given)
    ratings_valid = ratings[ratings['Book-Rating'] > 0].copy()
    print(f"Valid ratings count (excluding 0): {len(ratings_valid)}")
    
    # Merge ratings with book data to get book info and calculate popularity
    ratings_with_books = ratings_valid.merge(books[['ISBN', 'Book-Title', 'Book-Author']], 
                                             left_on='ISBN', right_on='ISBN', how='left')
    
    # Calculate book statistics for each ISBN
    book_stats = ratings_valid.groupby('ISBN').agg({
        'Book-Rating': ['mean', 'count', 'std']
    }).reset_index()
    book_stats.columns = ['ISBN', 'Rating-Mean', 'Rating-Count', 'Rating-Std']
    book_stats['Rating-Std'] = book_stats['Rating-Std'].fillna(0)
    
    print(f"Books with ratings: {len(book_stats)}")
    
    # Merge book stats back to books
    books = books.merge(book_stats, on='ISBN', how='left')
    books['Rating-Mean'] = books['Rating-Mean'].fillna(0)
    books['Rating-Count'] = books['Rating-Count'].fillna(0).astype(int)
    books['Rating-Std'] = books['Rating-Std'].fillna(0)
    
    # Normalize ratings (0-10 scale)
    books['Rating-Normalized'] = books['Rating-Mean'].clip(0, 10)
    
    # Calculate popularity score (combination of average rating and number of ratings)
    # Using log scale to not over-weight very popular books
    books['Popularity-Score'] = (books['Rating-Normalized'] * 0.7) + \
                                (np.log1p(books['Rating-Count']) * 0.3)
    
    print(f"Processed books count: {len(books)}")
    
    # Save processed data
    output_path = os.path.join('data', 'processed', 'cleaned_books.csv')
    books.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")
    
    # Save ratings for collaborative filtering
    ratings_output_path = os.path.join('data', 'processed', 'ratings_processed.csv')
    ratings_valid.to_csv(ratings_output_path, index=False)
    print(f"Saved processed ratings to {ratings_output_path}")
    
    # Save users for demographic analysis
    users_output_path = os.path.join('data', 'processed', 'users_processed.csv')
    users.to_csv(users_output_path, index=False)
    print(f"Saved processed users to {users_output_path}")

if __name__ == "__main__":
    preprocess_data()
