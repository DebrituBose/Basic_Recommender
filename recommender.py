import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Smart Recommender", page_icon="💡", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    body {
        background-color: #f9fafb;
    }
    .main {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    h1 {
        color: #4f46e5;
        text-align: center;
        font-weight: 700;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4f46e5, #06b6d4);
        color: white;
        font-size: 16px;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #06b6d4, #4f46e5);
        color: white;
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.title("💡 Smart Recommendation System")
st.markdown("Get personalized recommendations for movies, books, songs, clothes, and more — powered by AI 💫")

# ---------- DATA LOADING ----------
@st.cache_data
def load_data():
    books = pd.read_csv("books_small.csv")
    movies = pd.read_csv("movies_small.csv")
    songs = pd.read_csv("Spotify_small.csv")
    electronics = pd.read_csv("electronics_small.csv")
    foods = pd.read_csv("foods_small.csv")
    clothes = pd.read_csv("clothes_small.csv")
    return books, movies, songs, electronics, foods, clothes

books, movies, songs, electronics, foods, clothes = load_data()

# ---------- CATEGORY SELECTION ----------
st.sidebar.title("⚙️ Options")
category = st.sidebar.selectbox(
    "Choose a category",
    ["Books", "Movies", "Songs", "Electronics", "Foods", "Clothes"]
)

# ---------- FUNCTION TO RECOMMEND ----------
def get_recommendations(data, keyword_col, keywords):
    if keywords.strip() == "":
        return []
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(data[keyword_col].fillna(''))
    cosine_sim = cosine_similarity(tfidf.transform([keywords]), matrix)
    scores = list(enumerate(cosine_sim[0]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_items = [data.iloc[i[0]] for i in scores[:5] if i[1] > 0]
    return top_items

# ---------- INPUT SECTION ----------
st.markdown("### ✨ Enter your interests or keywords below:")
keywords = st.text_input(f"Enter keywords for {category} recommendation:")

if st.button("🔍 Get Recommendations"):
    if category == "Books":
        results = get_recommendations(books, "Description", keywords)
    elif category == "Movies":
        results = get_recommendations(movies, "overview", keywords)
    elif category == "Songs":
        results = get_recommendations(songs, "genre", keywords)
    elif category == "Electronics":
        results = get_recommendations(electronics, "Description", keywords)
    elif category == "Foods":
        results = get_recommendations(foods, "Description", keywords)
    elif category == "Clothes":
        results = get_recommendations(clothes, "Description", keywords)
    else:
        results = []

    if len(results) > 0:
        st.success(f"✅ Found {len(results)} matching recommendations!")
        for item in results:
            st.markdown(f"""
            <div style='background-color:#f1f5f9;padding:15px;border-radius:10px;margin-bottom:10px;'>
            <strong>{item.get('Name', 'Unnamed Item')}</strong><br>
            <span style='color:#555;'>{item.get('Description', 'No description available')}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("😔 No perfect match found. Try another keyword!")

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("<p style='text-align:center;color:gray;'>Made with ❤️ using Streamlit and scikit-learn</p>", unsafe_allow_html=True)
