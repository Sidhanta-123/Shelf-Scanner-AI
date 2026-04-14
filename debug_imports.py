
import sys
import os

packages = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "sqlalchemy",
    "sklearn",
    "pandas",
    "numpy",
    "cv2",
    "ultralytics",
    "pytesseract",
    "easyocr"
]

for pkg in packages:
    try:
        print(f"Testing {pkg}...", end=" ")
        __import__(pkg)
        print("OK")
    except ImportError as e:
        print(f"FAILED: {e}")
    except Exception as e:
        print(f"ERROR: {e}")
