import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Basic Recommender System", page_icon="🎧", layout="wide")

# ---- PAGE STYLE ----
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #89f7fe, #66a6ff);
    }
    .main {
        background: rgba(255,255,255,0.8);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    h1 {
        text-align:center;
        color:#1a1a1a;
        font-weight:700;
        margin-bottom:0;
    }
    h3 {
        text-align:center;
        color:#444;
    }
    .stTextInput input {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---- HEADER ----
st.markdown("<h1>🎧 Basic Recommender System</h1>", unsafe_allow_html=True)
st.markdown("<h3>Find books, movies, or songs similar to your choice 🎯</h3>", unsafe_allow_html=True)
st.markdown("---")

# ---- LOAD DATA ----
@st.cache_data
def load_data():
    books = pd.read_csv("books_small.csv")
    movies = pd.read_csv("movies_small.csv")
    songs = pd.read_csv("Spotify_small.csv")
    electronics = pd.read_csv("electronics_small.csv")
    foods = pd.read_csv("foods_small.csv")
    clothes = pd.read_csv("clothes_small.csv")
    return books, movies, songs, electronics, foods, clothes

try:
    books, movies, songs, electronics, foods, clothes = load_data()
except Exception as e:
    st.error("⚠️ Error loading one or more CSV files. Please check that all dataset files exist.")
    st.stop()

# ---- FUNCTION: RECOMMENDATION ----
def get_recommendations(data, keywords, category):
    """Return top 5 recommendations using TF-IDF cosine similarity."""
    if keywords.strip() == "":
        return []

    # Choose columns based on category
    if category == "Songs":
        text_cols = ['track_name', 'artist_name', 'genre']
    elif category == "Movies":
        text_cols = ['title', 'genres', 'overview']
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

    cosine_sim = cosine_similarity(tfidf.transform([keywords]), matrix)
    scores = list(enumerate(cosine_sim[0]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_items = [data.iloc[i[0]] for i in scores[:5] if i[1] > 0]
    return top_items

# ---- UI INPUTS ----
col1, col2 = st.columns([3, 1])
with col1:
    keywords = st.text_input("🔍 Enter keyword (e.g., *Shape of You*, *Harry Potter*, *Action Movie*):")
with col2:
    category = st.selectbox("📂 Choose Category", ["Books", "Movies", "Songs", "Electronics", "Foods", "Clothes"])

# ---- BUTTON ----
if st.button("✨ Get Recommendations"):
    if category == "Books":
        results = get_recommendations(books, keywords, category)
    elif category == "Movies":
        results = get_recommendations(movies, keywords, category)
    elif category == "Songs":
        results = get_recommendations(songs, keywords, category)
    elif category == "Electronics":
        results = get_recommendations(electronics, keywords, category)
    elif category == "Foods":
        results = get_recommendations(foods, keywords, category)
    elif category == "Clothes":
        results = get_recommendations(clothes, keywords, category)
    else:
        results = []

    if len(results) > 0:
        st.success(f"✅ Found {len(results)} matching recommendations!")
        for _, item in enumerate(results):
            st.markdown(f"""
            <div style='background-color:#f1f5f9;padding:15px;border-radius:15px;margin-bottom:10px;'>
                <strong>{item.get('track_name', item.get('title', item.get('Name', 'Unnamed Item')))}</strong><br>
                <em>{item.get('artist_name', item.get('Author_
