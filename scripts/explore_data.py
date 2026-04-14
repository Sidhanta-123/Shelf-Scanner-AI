import pandas as pd
import os

def check_csv(filename):
    print(f"\n--- {filename} ---")
    try:
        path = os.path.join('archive', filename)
        # Using on_bad_lines='skip' for robustness
        df = pd.read_csv(path, nrows=5, on_bad_lines='skip', low_memory=False)
        print("Columns:", df.columns.tolist())
        print(df.head())
    except Exception as e:
        print(f"Error reading {filename}: {e}")

if __name__ == "__main__":
    for f in ['Books.csv', 'Ratings.csv', 'Users.csv']:
        check_csv(f)
