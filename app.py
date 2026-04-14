import streamlit as st
from utils import get_book_data, generate_summary, generate_tags, calculate_recommendation_score
from utils import recommend_books

st.title("📚 Shelf Scanner AI")

# User input
book_title = st.text_input("Enter a book title")

if st.button("Search Book"):

    if book_title:
        book = get_book_data(book_title)

        if book:
            title = book.get("title", "Not found")
            author = book.get("author", "Not found")
            category = book.get("category", "Not found")
            rating = book.get("rating", "Not rated")
            description = book.get("description", "No description available.")

            st.subheader(title)
            st.write(f"👤 Author: {author if author else 'Unknown'}")
            st.write(f"🏷️ Category: {category if category else 'Unknown'}")
            st.write(f"⭐ Rating: {rating if rating else 'Not rated'}")

            st.write("📚 Description:")
            st.write(description)

            # AI Summary
            summary = generate_summary(description)

            st.subheader("📖 Summary")
            st.write(summary)

            # Tags 
            tags = generate_tags(description)

            st.subheader("🏷️ Tags")

            for tag in tags:
                st.write(f"• {tag}")

            # Recomendation Score
            score = calculate_recommendation_score(rating, description, category)

            st.subheader("⭐ AI Recommendation Score")
            st.write(f"{score} / 10")

            # Book Recommendation
            recommendations = recommend_books(description)

            st.subheader("📚 Similar Books You May Like")

            for book in recommendations:
                st.write(f"• {book}")

        else:
            st.write("❌ Book not found")
    else:
        st.write("⚠️ Please enter a book title")

