import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Basic Recommender System", layout="centered")

# ---------- STYLING ----------
st.markdown("""
<style>
body {background-color: #f0f4f8; color: #333333;}
.main {background-color: #ffffff; border-radius: 12px; padding: 25px; box-shadow: 0 0 12px rgba(0, 0, 0, 0.1);}
.title {text-align: center; color: #0078ff; font-size: 32px; font-weight: 700; margin-bottom: 10px;}
.footer {text-align: center; font-size: 13px; margin-top: 40px; color: #777;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">✨ Basic Recommender System ✨</div>', unsafe_allow_html=True)
st.write("Search across **Books, Movies, Songs, Clothes, Food, Products** to find items similar to your interest!")

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    datasets = {}
    try:
        datasets['Books'] = pd.read_csv("books_small.csv", low_memory=False)
        datasets['Movies'] = pd.read_csv("movies_small.csv", low_memory=False)
        datasets['Songs'] = pd.read_csv("Spotify_small.csv", low_memory=False)
        datasets['Clothes'] = pd.read_csv("clothes_small.csv", low_memory=False)
        datasets['Food'] = pd.read_csv("foods_small.csv", low_memory=False)
        datasets['Products'] = pd.read_csv("electronics_small.csv", low_memory=False)

        # Clean column names: strip spaces, replace spaces & hyphens with underscores
        for key in datasets:
            df = datasets[key]
            df.columns = [c.strip().replace(" ", "_").replace("-", "_") for c in df.columns]
            datasets[key] = df
        return datasets
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return {}

data_map = load_data()
categories = list(data_map.keys())

# ---------- RECOMMENDER FUNCTION ----------
def get_recommendations(data, keywords):
    if data is None or len(data) == 0 or keywords.strip() == "":
        return pd.DataFrame()

    # Exclude irrelevant columns for all datasets
    exclude_cols = [
        "Gender", "Marital_Status", "ID", "Price", 
        "Occupation", "Status", "Age", "Email", "Phone"
    ]
    text_cols = [c for c in data.columns if data[c].dtype == 'object' and c not in exclude_cols]

    if not text_cols:
        return pd.DataFrame()

    # Combine text for TF-IDF
    data["combined_text"] = data[text_cols].fillna("").astype(str).agg(" ".join, axis=1)

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(data["combined_text"])
    query_vec = tfidf.transform([keywords])
    cosine_sim = cosine_similarity(query_vec, matrix).flatten()

    # Get top 5 matches
    top_indices = cosine_sim.argsort()[-5:][::-1]
    top_scores = cosine_sim[top_indices]

    results = pd.DataFrame([data.iloc[idx] for idx, score in zip(top_indices, top_scores) if score > 0.01])

    # Fallback: random selection
    if results.empty:
        results = data.sample(min(5, len(data)))

    return results

# ---------- APP INTERFACE ----------
category = st.selectbox("Select Category:", categories)
keywords = st.text_input("Enter keywords (e.g., romance, thriller, dance, love, shirt, biryani, laptop):")

if st.button("🔍 Recommend"):
    with st.spinner("Finding recommendations..."):
        data = data_map.get(category, None)
        results = get_recommendations(data, keywords)

        if not results.empty:
            st.success("✅ Top Recommendations for you!")

            # Automatic display column selection
            exclude_cols = [
                "Gender", "Marital_Status", "ID", "Price", 
                "Occupation", "Status", "Age", "Email", "Phone"
            ]
            obj_cols = [c for c in results.select_dtypes(include='object').columns if c not in exclude_cols]
            display_col = obj_cols[0] if obj_cols else results.columns[0]

            for i, row in results.iterrows():
                value = row.get(display_col, "N/A")
                st.markdown(f"**{i+1}.** {value}")
        else:
            st.warning("😔 No results found. Try a different keyword!")

st.markdown('<div class="footer">Developed with ❤️ using Streamlit</div>', unsafe_allow_html=True)
