import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Basic Recommender System", layout="centered")

# ---------- STYLING ----------
st.markdown(
    """
    <style>
    body {
        background-color: #f0f4f8;
        color: #333333;
    }
    .main {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 0 12px rgba(0, 0, 0, 0.1);
    }
    .title {
        text-align: center;
        color: #0078ff;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .footer {
        text-align: center;
        font-size: 13px;
        margin-top: 40px;
        color: #777;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- PAGE HEADER ----------
st.markdown('<div class="title">✨ Basic Recommender System ✨</div>', unsafe_allow_html=True)
st.write("Search across **Books**, **Movies**, and **Songs** to find items similar to your interest!")

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    try:
        books = pd.read_csv("books.csv", low_memory=False)
        movies = pd.read_csv("tmdb_5000_movies.csv", low_memory=False)
        songs = pd.read_csv("SpotifyFeatures.csv", low_memory=False)
        return books, movies, songs
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

books, movies, songs = load_data()

# ---------- IMPROVED RECOMMENDER FUNCTION ----------
def get_recommendations(data, keywords, category):
    """Return top 5 recommendations using TF-IDF cosine similarity."""
    if data is None or len(data) == 0 or keywords.strip() == "":
        return []

    # Choose text columns based on category
    if category == "Songs":
        text_cols = ['track_name', 'artist_name', 'genre']
    elif category == "Movies":
        text_cols = ['title', 'genres', 'overview']
    elif category == "Books":
        text_cols = ['Name', 'Book-Title', 'Author', 'Description']
    else:
        text_cols = [c for c in data.columns if data[c].dtype == 'object']

    # Add missing columns
    for col in text_cols:
        if col not in data.columns:
            data[col] = ""

    # Combine text from multiple columns
    data["combined_text"] = data[text_cols].fillna("").astype(str).agg(" ".join, axis=1)

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(data["combined_text"])

    # Compute similarity with query
    query_vec = tfidf.transform([keywords])
    cosine_sim = cosine_similarity(query_vec, matrix).flatten()

    # Get top matches
    top_indices = cosine_sim.argsort()[-5:][::-1]
    top_scores = cosine_sim[top_indices]

    # Filter out weak matches
    results = []
    for idx, score in zip(top_indices, top_scores):
        if score > 0.01:
            results.append(data.iloc[idx])

    # Fallback: random if nothing matches
    if len(results) == 0:
        results = data.sample(min(5, len(data)))

    return results

# ---------- APP INTERFACE ----------
category = st.radio("Select Category:", ["Books", "Movies", "Songs"])
keywords = st.text_input("Enter keywords (e.g., romance, thriller, dance, love):")

if st.button("🔍 Recommend"):
    with st.spinner("Finding recommendations..."):
        if category == "Books":
            data = books
        elif category == "Movies":
            data = movies
        elif category == "Songs":
            data = songs
        else:
            data = None

        results = get_recommendations(data, keywords, category)

        if len(results) > 0:
            st.success("✅ Top Recommendations for you!")
            for i, row in enumerate(results, 1):
                st.markdown(f"**{i}.** {row.iloc[0]}")
        else:
            st.warning("😔 No results found. Try a different keyword!")

# ---------- FOOTER ----------
st.markdown('<div class="footer">Developed with ❤️ using Streamlit</div>', unsafe_allow_html=True)
