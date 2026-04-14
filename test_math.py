import re
from Levenshtein import ratio as lev_ratio

query = "Easy Way"
full_ocr = "TI Changes Remarkable Results Atoric Habits An Proven OVER MILLION to Build Good Habits COPIES SOLD Break Bad Ones James Clear Tiny Easy Way"
full_text_lower = (full_ocr + " " + query).lower()
q_lower = query.lower()

title = "Way of the Peaceful Warrior".lower()
authors = ["Dan Millman"]

title_score = lev_ratio(q_lower, title)
print(f"Base Lev Ratio: {title_score}")

if q_lower in title or title in q_lower:
    title_score = max(title_score, 0.8)
    print(f"Substring Match! Score is now {title_score}")

if len(title) > 4 and title in full_text_lower:
    title_score += 0.3
    print(f"Title in full text! Score is now {title_score}")

author_bonus = 0.0
for author in authors:
    author = author.lower()
    if len(author) > 3 and author in full_text_lower:
        author_bonus = 0.4
        print(f"Author exact match! Bonus 0.4")
        break
    elif len(author) > 3:
        for part in author.split():
            if len(part) > 3 and part in full_text_lower:
                author_bonus = 0.2
                print(f"Author partial match! Bonus 0.2 for {part}")
                break

score = title_score + author_bonus
print(f"Total Score: {score}")
