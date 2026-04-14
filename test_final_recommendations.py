#!/usr/bin/env python
"""Test the updated recommendation engine with exclude_title parameter."""
import requests
import json

# Test 1: Search for The Hobbit
print('='*60)
print('TEST 1: Searching for "The Hobbit"')
print('='*60)
r = requests.post('http://127.0.0.1:8000/lookup', json={'query': 'The Hobbit'})
book_data = r.json()['book']
similar = book_data.get('similar_books', [])
print(f'Book Found: {book_data["title"]}')
print('Recommendations:')
for i, rec in enumerate(similar[:3], 1):
    score = rec['similarity_score'] * 100
    print(f'{i}. {rec["title"]:<40} | {score:>3.0f}% match')

# Test 2: Search for Atomic Habits
print()
print('='*60)
print('TEST 2: Searching for "Atomic Habits"')
print('='*60)
r = requests.post('http://127.0.0.1:8000/lookup', json={'query': 'Atomic Habits'})
book_data = r.json()['book']
similar = book_data.get('similar_books', [])
print(f'Book Found: {book_data["title"]}')
print('Recommendations:')
for i, rec in enumerate(similar[:3], 1):
    score = rec['similarity_score'] * 100
    print(f'{i}. {rec["title"]:<40} | {score:>3.0f}% match')

# Test 3: Search for Crime Mystery
print()
print('='*60)
print('TEST 3: Searching for "Crime Mystery Detective"')
print('='*60)
r = requests.post('http://127.0.0.1:8000/lookup', json={'query': 'Crime Mystery Detective'})
book_data = r.json()['book']
similar = book_data.get('similar_books', [])
print(f'Book Found: {book_data["title"]}')
print('Recommendations:')
for i, rec in enumerate(similar[:3], 1):
    score = rec['similarity_score'] * 100
    print(f'{i}. {rec["title"]:<40} | {score:>3.0f}% match')
