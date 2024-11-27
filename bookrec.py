import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import re

# Load the dataset
file_path = r"unique_books.csv"  # Replace with the correct path if necessary
books_df = pd.read_csv(file_path)

# Inspect column names
print("Original Columns in the Dataset:", books_df.columns)

# Rename columns to ensure consistent naming
column_mapping = {
    'title': 'Title',  # Replace 'title' with 'Title' if case mismatch exists
    'description': 'Description',  # Replace 'description' with 'Description'
}
books_df.rename(columns=column_mapping, inplace=True)

# Check for missing columns
required_columns = ['Title', 'Description']
for col in required_columns:
    if col not in books_df.columns:
        raise ValueError(f"Missing required column: {col}")

# Fill missing values in the 'Description' column
books_df['Description'] = books_df['Description'].fillna('')

# Preprocess text in the 'Description' column
def preprocess_text(text):
    return re.sub(r'[^a-z\s]', '', str(text).lower())

books_df['processed_description'] = books_df['Description'].apply(preprocess_text)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(books_df['processed_description'])

# Train the kNN model
n_neighbors = 6  # Number of neighbors to return
model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
model.fit(tfidf_matrix)

# Create a mapping from book titles to their indices
title_to_index = pd.Series(books_df.index, index=books_df['Title'].str.lower())

# Recommendation function
def recommend_books(title, model=model, books_df=books_df, title_to_index=title_to_index):
    try:
        idx = title_to_index[title.lower()]
        distances, indices = model.kneighbors(tfidf_matrix[idx])
        indices = indices[0][1:]  # Exclude the input book itself
        return books_df['Title'].iloc[indices].values.tolist()
    except KeyError:
        return []

# Main program
def main():
    while True:
        book_title = input("Enter the book title (Type 'end' to end the program): ").strip()
        if book_title.lower() == 'end':
            print("Exiting program. Goodbye!")
            break
        if book_title.lower() not in title_to_index:
            print("Sorry! Book not found in the database! Try another book!")
            continue

        recommendations = recommend_books(book_title)
        if recommendations:
            print("\nThe following are the best recommendations for '{}':\n".format(book_title))
            for rec in recommendations:
                print("- " + rec)
        else:
            print("Sorry, no recommendations found!")
        print("\n")

if __name__ == "__main__":
    main()
