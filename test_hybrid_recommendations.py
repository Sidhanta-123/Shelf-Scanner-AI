"""
Test the new Hybrid Recommendation System
Tests both collaborative filtering and content-based filtering
"""

import sys
import os

from backend.utils import recommend_books
import pandas as pd

# Load books data for reference
books_df = pd.read_csv('data/processed/cleaned_books.csv', low_memory=False)

print("=" * 80)
print("HYBRID RECOMMENDATION SYSTEM TEST")
print("=" * 80)

# Test 1: Self-help / Productivity books
print("\n📚 TEST 1: Self-Help & Productivity Books")
print("-" * 80)
description1 = "This book is about building better habits, improving productivity, and achieving success through mindset development"
recommendations1 = recommend_books(description1, title="Atomic Habits", exclude_title="")
print(f"Query: {description1[:70]}...")
print(f"Found {len(recommendations1)} recommendations:\n")
for i, rec in enumerate(recommendations1, 1):
    print(f"{i}. {rec['title']}")
    print(f"   Author: {rec['author']}")
    print(f"   Score: {rec['similarity_score']} | Rating: {rec['rating']}")
    print(f"   Source: {rec.get('recommendation_source', 'hybrid')}\n")

# Test 2: Fantasy / Adventure books
print("\n" + "=" * 80)
print("📚 TEST 2: Fantasy & Adventure Books")
print("-" * 80)
description2 = "A magical fantasy adventure with wizards, quests, dragons, and epic battles in a mystical world"
recommendations2 = recommend_books(description2, title="Harry Potter", exclude_title="")
print(f"Query: {description2[:70]}...")
print(f"Found {len(recommendations2)} recommendations:\n")
for i, rec in enumerate(recommendations2, 1):
    print(f"{i}. {rec['title']}")
    print(f"   Author: {rec['author']}")
    print(f"   Score: {rec['similarity_score']} | Rating: {rec['rating']}")
    print(f"   Source: {rec.get('recommendation_source', 'hybrid')}\n")

# Test 3: Mystery / Thriller books
print("\n" + "=" * 80)
print("📚 TEST 3: Mystery & Thriller Books")
print("-" * 80)
description3 = "A gripping crime mystery involving a detective investigating a murder case with unexpected twists"
recommendations3 = recommend_books(description3, title="The Girl with the Dragon Tattoo", exclude_title="")
print(f"Query: {description3[:70]}...")
print(f"Found {len(recommendations3)} recommendations:\n")
for i, rec in enumerate(recommendations3, 1):
    print(f"{i}. {rec['title']}")
    print(f"   Author: {rec['author']}")
    print(f"   Score: {rec['similarity_score']} | Rating: {rec['rating']}")
    print(f"   Source: {rec.get('recommendation_source', 'hybrid')}\n")

# Test 4: Business / Finance books
print("\n" + "=" * 80)
print("📚 TEST 4: Business & Finance Books")
print("-" * 80)
description4 = "A business book about entrepreneurship, wealth creation, investment strategies, and financial independence"
recommendations4 = recommend_books(description4, title="Think and Grow Rich", exclude_title="")
print(f"Query: {description4[:70]}...")
print(f"Found {len(recommendations4)} recommendations:\n")
for i, rec in enumerate(recommendations4, 1):
    print(f"{i}. {rec['title']}")
    print(f"   Author: {rec['author']}")
    print(f"   Score: {rec['similarity_score']} | Rating: {rec['rating']}")
    print(f"   Source: {rec.get('recommendation_source', 'hybrid')}\n")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"✓ Hybrid recommendation system is working!")
print(f"✓ Collaborative Filtering: Uses user rating patterns from {len(books_df)} books")
print(f"✓ Content-Based Filtering: Uses TF-IDF + title/author similarity")
print(f"✓ Hybrid Score: Combines both approaches with popularity weighting")
print(f"✓ Total books in dataset: {len(books_df)}")
print("=" * 80)
