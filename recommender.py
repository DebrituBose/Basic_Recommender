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

# ---------- LOAD FILES WITH CSV AND MALFORMED ROW FIX ----------
def read_file(file_path):
    """Read CSV, skip bad lines, handle encoding."""
    try:
        return pd.read_csv(file_path, encoding="utf-8", on_bad_lines='skip', low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="ISO-8859-1", on_bad_lines='skip', low_memory=False)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return pd.DataFrame()

@st.cache_data
def load_data():
    food = read_file("foods.csv")
    clothes = read_file("clothes.csv")
    products = read_file("products.csv")
    movies = read_file("movie.csv")
    songs = read_file("Spotify_small.csv")
    books = read_file("books_small.csv")

    # ---------- CLEAN FOOD DATA ----------
    if not food.empty:
        # Strip whitespace from column names
        food.columns = food.columns.str.strip()
        # Automatically detect Name and Restaurant columns
        name_col = [c for c in food.columns if 'Name' in c.lower()][0]  # first column containing 'name'
        rest_col = [c for c in food.columns if 'Restaurant' in c.lower()][0]  # first containing 'restaurant'
        # Keep only valid rows
        food = food[food[Name_col].notna() & food[Rest_col].notna()]
        # Strip whitespace from relevant columns
        text_cols = [Name_col, Rest_col, 'Category', 'Description']
        for col in text_cols:
            if col in food.columns:
                food[col] = food[col].astype(str).str.strip()
        # Save detected column names for later display
        food._Name_col = name_col
        food._Rest_col = rest_col

    return food, clothes, products, movies, songs, books

food, clothes, products, movies, songs, books = load_data()

# ---------- RECOMMENDER FUNCTION ----------
def get_recommendations(data, keywords, category):
    if data is None or data.empty or keywords.strip() == "":
        return []

    # Select text columns
    if category == "Food":
        text_cols = [data._Name_col, data._Rest_col, 'Category', 'Description']
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

    for col in text_cols:
        if col not in data.columns:
            data[col] = ""

    data["combined_text"] = data[text_cols].fillna("").astype(str).agg(" ".join, axis=1)

    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(data["combined_text"])

    query_vec = tfidf.transform([keywords])
    cosine_sim = cosine_similarity(query_vec, matrix).flatten()

    top_indices = cosine_sim.argsort()[-5:][::-1]
    top_scores = cosine_sim[top_indices]

    results = []
    for idx, score in zip(top_indices, top_scores):
        if score > 0.01:
            results.append(data.iloc[idx])

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
            display_cols = [data._name_col, data._rest_col, 'Category', 'Price', 'Description']
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
                info = " | ".join([f"{col}: {row[col]}" for col in display_cols if col in row])
                st.markdown(f"**{i}.** {info}")
        else:
            st.warning("😔 No results found. Try a different keyword!")

# ---------- FOOTER ----------
st.markdown('<div class="footer">Developed with ❤️ using Streamlit by Debritu Bose</div>', unsafe_allow_html=True)

