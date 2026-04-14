"""
Google Books API client — fetches rich book insights for the library companion use-case.
No API key required for basic queries (up to 1000 req/day per IP).
"""
import urllib.request
import urllib.parse
import json
import re


GOOGLE_BOOKS_API = "https://www.googleapis.com/books/v1/volumes"


def _fetch(url: str):
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"Google Books API error: {e}")
        return None


def search_book(query: str):
    """
    Search Google Books for a query string and return rich info for the top result.
    Returns a dict with keys: title, author, genre, description, rating, thumbnail,
    page_count, publisher, published_date, isbn, preview_link, similar_queries
    """
    if not query or len(query.strip()) < 2:
        return None

    encoded = urllib.parse.quote(query.strip())
    url = f"{GOOGLE_BOOKS_API}?q={encoded}&maxResults=5&langRestrict=en&printType=books"

    data = _fetch(url)
    if not data or data.get("totalItems", 0) == 0:
        return None

    items = data.get("items", [])
    if not items:
        return None

    # Take the first result
    item = items[0]
    info = item.get("volumeInfo", {})

    # Extract fields
    title = info.get("title", "Unknown Title")
    authors = info.get("authors", ["Unknown Author"])
    author = ", ".join(authors)
    publisher = info.get("publisher", "")
    published_date = info.get("publishedDate", "")
    description = info.get("description", "No description available.")
    categories = info.get("categories", [])
    genre = ", ".join(categories) if categories else "General / Non-Fiction"
    average_rating = info.get("averageRating", None)
    ratings_count = info.get("ratingsCount", 0)
    page_count = info.get("pageCount", None)
    preview_link = info.get("previewLink", "")
    info_link = info.get("infoLink", "")

    # Thumbnail
    image_links = info.get("imageLinks", {})
    thumbnail = (
        image_links.get("thumbnail")
        or image_links.get("smallThumbnail")
        or ""
    )
    # Force HTTPS
    thumbnail = thumbnail.replace("http://", "https://")

    # ISBN
    isbn = ""
    for identifier in info.get("industryIdentifiers", []):
        if identifier.get("type") in ("ISBN_13", "ISBN_10"):
            isbn = identifier.get("identifier", "")
            break

    # Collect other results as "similar books"
    similar = []
    for other in items[1:]:
        other_info = other.get("volumeInfo", {})
        other_thumb = other_info.get("imageLinks", {}).get("thumbnail", "").replace("http://", "https://")
        similar.append({
            "title": other_info.get("title", ""),
            "author": ", ".join(other_info.get("authors", [])),
            "thumbnail": other_thumb,
            "preview_link": other_info.get("previewLink", ""),
        })

    return {
        "title": title,
        "author": author,
        "publisher": publisher,
        "published_date": published_date,
        "description": description,
        "genre": genre,
        "average_rating": average_rating,
        "ratings_count": ratings_count,
        "page_count": page_count,
        "thumbnail": thumbnail,
        "isbn": isbn,
        "preview_link": preview_link,
        "info_link": info_link,
        "similar_books": similar,
    }
