from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import re

app = Flask(__name__)

# Load the dataset
books_df = pd.read_csv("unique_books.csv")  # Ensure the CSV is in the same folder as this script

# Preprocess text
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
def recommend_books(title):
    try:
        idx = title_to_index[title.lower()]
        distances, indices = model.kneighbors(tfidf_matrix[idx])
        indices = indices[0][1:]  # Exclude the input book itself
        return books_df['Title'].iloc[indices].values.tolist()
    except KeyError:
        return []

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    user_input = request.form.get("book_title", "").strip()
    if not user_input:
        return jsonify({"error": "Book title cannot be empty."})
    
    recommendations = recommend_books(user_input)
    if not recommendations:
        return jsonify({"error": "Book not found or no recommendations available."})
    
    return jsonify({"recommendations": recommendations})

if __name__ == "__main__":
    app.run(debug=True)
