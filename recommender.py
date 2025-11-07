import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Universal Recommender System", layout="centered")

# ---------- STYLING ----------
st.markdown("""
<style>
body { background-color: #f0f4f8; color: #333333; }
.main { background-color: #ffffff; border-radius: 12px; padding: 25px; box-shadow: 0 0 12px rgba(0,0,0,0.1); }
.title { text-align: center; color: #0078ff; font-size: 32px; font-weight: 700; margin-bottom: 10px; }
.footer { text-align: center; font-size: 13px; margin-top: 40px; color: #777; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">✨ Universal Recommender System ✨</div>', unsafe_allow_html=True)
st.write("Search across **Food**, **Clothes**, **Products**, **Movies**, **Songs**, and **Books**!")

# ---------- FILE UPLOAD ----------
def read_file(uploaded_file):
    if uploaded_file is None:
        return pd.DataFrame()
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file, encoding="utf-8", on_bad_lines='skip', low_memory=False)
        elif name.endswith(".xlsx"):
            return pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type! Upload .csv or .xlsx")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return pd.DataFrame()

st.sidebar.header("Upload your datasets")
food_file = st.sidebar.file_uploader("Upload Food CSV/XLSX", type=["csv","xlsx"])
clothes_file = st.sidebar.file_uploader("Upload Clothes CSV/XLSX", type=["csv","xlsx"])
products_file = st.sidebar.file_uploader("Upload Products CSV/XLSX", type=["csv","xlsx"])
movies_file = st.sidebar.file_uploader("Upload Movies CSV/XLSX", type=["csv","xlsx"])
songs_file = st.sidebar.file_uploader("Upload Songs CSV/XLSX", type=["csv","xlsx"])
books_file = st.sidebar.file_uploader("Upload Books CSV/XLSX", type=["csv","xlsx"])

@st.cache_data
def load_data():
    food = read_file(foods_file)
    clothes = read_file(clothes_file)
    products = read_file(products_file)
    movies = read_file(movie_file)
    songs = read_file(Spotify_small_file)
    books = read_file(books_small_file)

    # ---------- CLEAN FOOD DATA ----------
    if not food.empty:
        food.columns = food.columns.str.strip().str.lower()
        st.write("Food CSV Columns Detected:", food.columns.tolist())
        if 'name' in food.columns and 'restaurant' in food.columns:
            food = food[food['name'].notna() & food['restaurant'].notna()]
            for col in ['name','restaurant','category','description']:
                if col in food.columns:
                    food[col] = food[col].astype(str).str.strip()
        else:
            st.error("Food CSV must have columns 'Name' and 'Restaurant' (any case)!")

    return food, clothes, products, movies, songs, books

food, clothes, products, movies, songs, books = load_data()

# ---------- RECOMMENDER FUNCTION ----------
def get_recommendations(data, keywords, category):
    if data is None or data.empty or keywords.strip() == "":
        return []

    # select text columns for similarity
    if category == "Food":
        text_cols = ['name','restaurant','category','description']
    elif category == "Clothes":
        text_cols = ['Name','Brand','Category','Description']
    elif category == "Products":
        text_cols = ['Name','Brand','Category','Description']
    elif category == "Movies":
        text_cols = ['title','genres','overview']
    elif category == "Songs":
        text_cols = ['track_name','artist_name','genre']
    elif category == "Books":
        text_cols = ['Name','Book-Title','Author','Description']
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
    ["Food","Clothes","Products","Movies","Songs","Books"]
)
keywords = st.text_input("Enter keywords (e.g., biryani, jeans, laptop, action, love, dance):")

if st.button("🔍 Recommend"):
    with st.spinner("Finding recommendations..."):
        if category == "Food":
            data = food
            display_cols = ['name','restaurant','category','price','description']
        elif category == "Clothes":
            data = clothes
            display_cols = ['Name','Brand','Category','Price','Description']
        elif category == "Products":
            data = products
            display_cols = ['Name','Brand','Category','Price','Description']
        elif category == "Movies":
            data = movies
            display_cols = ['title','genres','overview']
        elif category == "Songs":
            data = songs
            display_cols = ['track_name','artist_name','genre']
        elif category == "Books":
            data = books
            display_cols = ['Name','Book-Title','Author','Description']
        else:
            data = None
            display_cols = []

        results = get_recommendations(data, keywords, category)

        if len(results) > 0:
            st.success("✅ Top Recommendations for you!")
            for i, row in enumerate(results,1):
                info = " | ".join([f"{col}: {row[col]}" for col in display_cols if col in row])
                st.markdown(f"**{i}.** {info}")
        else:
            st.warning("😔 No results found. Try a different keyword!")

st.markdown('<div class="footer">Developed with ❤️ using Streamlit</div>', unsafe_allow_html=True)
