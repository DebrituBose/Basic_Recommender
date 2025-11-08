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
    food = read_file(food_file)
    clothes = read_file(clothes_file)
    products = read_file(products_file)
    movies = read_file(movies_file)
    songs = read_file(songs_file)
    books = read_file(books_file)

    # ---------- CLEAN DATA ----------
    def clean_df(df, required_cols=[]):
        if df.empty:
            return df
        # Normalize column names: lowercase, remove spaces, replace '-' with '_'
        df.columns = [c.strip().lower().replace(" ","_").replace("-","_") for c in df.columns]
        # Add missing columns
        for col in required_cols:
            col_lower = col.strip().lower().replace(" ","_").replace("-","_")
            if col_lower not in df.columns:
                df[col_lower] = ""
        # Strip text
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype(str).str.strip()
        return df

    food = clean_df(food, ['name','restaurant','category','description','price'])
    clothes = clean_df(clothes, ['name','brand','category','description','price'])
    products = clean_df(products, ['name','brand','category','description','price'])
    movies = clean_df(movies, ['title','genres','overview'])
    songs = clean_df(songs, ['track_name','artist_name','genre'])
    books = clean_df(books, ['name','book_title','author','description'])

    return food, clothes, products, movies, songs, books

food, clothes, products, movies, songs, books = load_data()

# ---------- RECOMMENDER FUNCTION ----------
def get_recommendations(data, keywords, category):
    if data.empty or not keywords.strip():
        return []

    text_cols_dict = {
        "Food": ['name','restaurant','category','description'],
        "Clothes": ['name','brand','category','description'],
        "Products": ['name','brand','category','description'],
        "Movies": ['title','genres','overview'],
        "Songs": ['track_name','artist_name','genre'],
        "Books": ['name','book_title','author','description']
    }

    text_cols = [c for c in text_cols_dict.get(category, []) if c in data.columns]
    if not text_cols:
        text_cols = [c for c in data.columns if data[c].dtype=='object']

    # Combine text safely
    data["combined_text"] = data[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    data["combined_text"] = data["combined_text"].replace(r'^\s*$', 'empty', regex=True)  # placeholder

    keywords = keywords.lower()

    try:
        tfidf = TfidfVectorizer(stop_words='english')
        matrix = tfidf.fit_transform(data["combined_text"])
        query_vec = tfidf.transform([keywords])
        cosine_sim = cosine_similarity(query_vec, matrix).flatten()

        top_indices = cosine_sim.argsort()[-5:][::-1]
        results = []
        for idx in top_indices:
            if cosine_sim[idx] > 0.01:
                results.append(data.iloc[idx])
        if len(results) == 0:
            results = data.sample(min(5, len(data))).to_dict('records')
            results = [pd.Series(r) for r in results]
        return results
    except ValueError:
        # fallback if TF-IDF fails
        results = data.sample(min(5, len(data))).to_dict('records')
        return [pd.Series(r) for r in results]

# ---------- APP INTERFACE ----------
category = st.radio(
    "Select Category:",
    ["Food","Clothes","Products","Movies","Songs","Books"]
)
keywords = st.text_input("Enter keywords (e.g., biryani, jeans, laptop, action, love, dance):")

if st.button("🔍 Recommend"):
    with st.spinner("Finding recommendations..."):
        data_dict = {
            "Food": food,
            "Clothes": clothes,
            "Products": products,
            "Movies": movies,
            "Songs": songs,
            "Books": books
        }

        display_cols_dict = {
            "Food": ['name','restaurant','category','price','description'],
            "Clothes": ['name','brand','category','price','description'],
            "Products": ['name','brand','category','price','description'],
            "Movies": ['title','genres','overview'],
            "Songs": ['track_name','artist_name','genre'],
            "Books": ['name','book_title','author','description']
        }

        data = data_dict.get(category, pd.DataFrame())
        display_cols = display_cols_dict.get(category, [])

        results = get_recommendations(data, keywords, category)

        if results:
            st.success("✅ Top Recommendations for you!")
            for i, row in enumerate(results,1):
                info_parts = []
                for col in display_cols:
                    val = row.get(col, "")
                    if pd.notna(val) and str(val).strip():
                        info_parts.append(f"{col.replace('_',' ').title()}: {val}")
                info = " | ".join(info_parts)
                st.markdown(f"**{i}.** {info}")
        else:
            st.warning("😔 No results found. Try a different keyword!")

st.markdown('<div class="footer">Developed with ❤️ using Streamlit</div>', unsafe_allow_html=True)
