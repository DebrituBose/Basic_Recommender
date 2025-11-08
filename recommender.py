import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Basic Recommender System", page_icon="✨", layout="centered")

st.markdown(
    "<h1 style='text-align: center; color: #007BFF;'>✨ Basic Recommender System ✨</h1>",
    unsafe_allow_html=True,
)

st.write("Search across **Food, Clothes, Products, Movies, Songs, and Books!**")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    def read_file(file_path):
        try:
            if file_path.endswith('.csv'):
                return pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
            else:
                st.warning(f"⚠ Unsupported file format: {file_path}")
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Error reading file {file_path}: {e}")
            return pd.DataFrame()

    food = read_file("foods.csv")
    clothes = read_file("clothes.csv")
    products = read_file("products.csv")
    movies = read_file("movies.csv")
    songs = read_file("songs.csv")
    books = read_file("books.csv")

    return food, clothes, products, movies, songs, books


food, clothes, products, movies, songs, books = load_data()

# ---------- DATA SUMMARY ----------
st.subheader("📊 Data Loaded Summary (for debugging)")
st.write("Food:", food.shape)
st.write("Clothes:", clothes.shape)
st.write("Products:", products.shape)
st.write("Movies:", movies.shape)
st.write("Songs:", songs.shape)
st.write("Books:", books.shape)
st.write("---")


# ---------------- PREPROCESS ----------------
def prepare_data(df, name_col="Name", desc_col="Description"):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if name_col not in df.columns:
        possible_name = [c for c in df.columns if "name" in c.lower()]
        if possible_name:
            name_col = possible_name[0]
        else:
            df[name_col] = "Unknown"

    if desc_col not in df.columns:
        possible_desc = [c for c in df.columns if "desc" in c.lower()]
        if possible_desc:
            desc_col = possible_desc[0]
        else:
            df[desc_col] = "No description available"

    df["combined_text"] = (
        df[name_col].astype(str) + " " + df[desc_col].astype(str)
    )
    return df


food = prepare_data(food)
clothes = prepare_data(clothes)
products = prepare_data(products)
movies = prepare_data(movies)
songs = prepare_data(songs)
books = prepare_data(books)


# ---------------- RECOMMENDER FUNCTION ----------------
def get_recommendations(df, query):
    if df.empty or "combined_text" not in df.columns:
        return []

    tfidf = TfidfVectorizer(stop_words="english")
    try:
        matrix = tfidf.fit_transform(df["combined_text"])
    except ValueError:
        return []

    query_vec = tfidf.transform([query])
    sim = cosine_similarity(query_vec, matrix).flatten()

    indices = sim.argsort()[-5:][::-1]
    results = df.iloc[indices]

    # Only return matches that have some similarity (ignore zeros)
    results = results[sim[indices] > 0]
    return results


# ---------------- UI SECTION ----------------
category = st.radio(
    "Select Category:",
    ("Food", "Clothes", "Products", "Movies", "Songs", "Books")
)

query = st.text_input("Enter keywords (e.g., biryani, jeans, laptop, action, love, dance):")
if st.button("🔍 Recommend"):
    category_map = {
        "Food": food,
        "Clothes": clothes,
        "Products": products,
        "Movies": movies,
        "Songs": songs,
        "Books": books,
    }
    df = category_map[category]

    if query.strip():
        results = get_recommendations(df, query.strip())
        if not results.empty:
            st.success(f"### Top {len(results)} {category} Recommendations:")
            for i, row in results.iterrows():
                name = str(row.get("Name", "Unknown"))
                desc = str(row.get("Description", ""))[:250]
                st.markdown(f"**{name}** — {desc}")
                st.write("---")
        else:
            st.warning("😔 No results found. Try a different keyword!")
    else:
        st.info("👉 Please enter a keyword to search.")


st.markdown("<br><center>Developed with ❤️ using Streamlit by <b>Debritu Bose</b></center>", unsafe_allow_html=True)
