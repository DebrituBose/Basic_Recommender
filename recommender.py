import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Basic Recommender System", layout="centered")

# ---------- STYLING ----------
st.markdown("""
<style>
body { background-color: #f0f4f8; color: #333333; }
.main { background-color: #ffffff; border-radius: 12px; padding: 25px; box-shadow: 0 0 12px rgba(0,0,0,0.1); }
.title { text-align: center; color: #0078ff; font-size: 32px; font-weight: 700; margin-bottom: 10px; }
.footer { text-align: center; font-size: 13px; margin-top: 40px; color: #777; }
</style>
""", unsafe_allow_html=True)

# ---------- PAGE HEADER ----------
st.markdown('<div class="title">✨ Basic Recommender System ✨</div>', unsafe_allow_html=True)
st.write("Search across **Food**, **Clothes**, **Products**, **Movies**, **Songs**, and **Books**!")

# ---------- LOAD CSV WITH ENCODING FALLBACK ----------
def read_csv_with_fallback(file_path):
    try:
        return pd.read_csv(file_path, low_memory=False, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(file_path, low_memory=False, encoding='ISO-8859-1')

@st.cache_data
def load_data():
    try:
        food = read_csv_with_fallback("food.csv.xlsx")
        clothes = read_csv_with_fallback("clothes.csv.xlsx")
        products = read_csv_with_fallback("products.csv.xlsx")
        movies = read_csv_with_fallback("Book1.xlsx")
        songs = read_csv_with_fallback("Spotify_small.csv")
        books = read_csv_with_fallback("books_small.csv")
        return food, clothes, products, movies, songs, books
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None, None

food, clothes, products, movies, songs, books = load_data()

# ---------- RECOMMENDER FUNCTION ----------
def get_recommendations(data, keywords, category):
    if data is None or len(data) == 0 or keywords.strip() == "":
        return []

    # Select text columns
    if category == "Food":
        text_cols = ['Name', 'Restaurant', 'Category', 'Description']
    elif category == "Clothes":
        text_cols = ['Name', 'Brand', 'Category', 'Description']
    elif category == "Products":
        text_cols = ['Name', 'Brand', 'Category', 'Description']
    elif category == "Movies":
        text_cols = ['title', 'genres', 'overview']
    elif category == "Songs":
        text_cols = ['track_name', 'artist_name', 'genre']
    elif category == "Books":
        text_cols = ['Name', 'Book-Title', 'Author', 'Description']
    else:
        text_cols = [c for c in data.columns if data[c].dtype == 'object']

    # Ensure all columns exist
    for col in text_cols:
        if col not in data.columns:
            data[col] = ""

    # Combine text
    data["combined_text"] = data[text_cols].fillna("").astype(str).agg(" ".join, axis=1)

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(data["combined_text"])

    # Compute similarity
    query_vec = tfidf.transform([keywords])
    cosine_sim = cosine_similarity(query_vec, matrix).flatten()

    # Get top matches
    top_indices = cosine_sim.argsort()[-5:][::-1]
    top_scores = cosine_sim[top_indices]

    results = []
    for idx, score in zip(top_indices, top_scores):
        if score > 0.01:
            results.append(data.iloc[idx])

    # Fallback random sample
    if len(results) == 0:
        results = data.sample(min(5, len(data)))

    return results

# ---------- APP INTERFACE ----------
category = st.radio(
    "Select Category:",
    ["Food", "Clothes", "Products", "Movies", "Songs", "Books"]
)
keywords = st.text_input("Enter keywords (e.g., biryani, jeans, laptop, action, love, dance):")

if st.button("🔍 Recommend"):
    with st.spinner("Finding recommendations..."):
        if category == "Food":
            data = food
            display_cols = ['Name', 'Restaurant', 'Category', 'Price', 'Description']
        elif category == "Clothes":
            data = clothes
            display_cols = ['Name', 'Brand', 'Category', 'Price', 'Description']
        elif category == "Products":
            data = products
            display_cols = ['Name', 'Brand', 'Category', 'Price', 'Description']
        elif category == "Movies":
            data = movies
            display_cols = ['title', 'genres', 'overview']
        elif category == "Songs":
            data = songs
            display_cols = ['track_name', 'artist_name', 'genre']
        elif category == "Books":
            data = books
            display_cols = ['Name', 'Book-Title', 'Author', 'Description']
        else:
            data = None
            display_cols = []

        results = get_recommendations(data, keywords, category)

        if len(results) > 0:
            st.success("✅ Top Recommendations for you!")
            for i, row in enumerate(results, 1):
                info = " | ".join([f"{col}: {row[col]}" for col in display_cols])
                st.markdown(f"**{i}.** {info}")
        else:
            st.warning("😔 No results found. Try a different keyword!")

# ---------- FOOTER ----------
st.markdown('<div class="footer">Developed with ❤️ using Streamlit by Debritu Bose</div>', unsafe_allow_html=True)
