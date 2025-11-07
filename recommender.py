import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Load datasets ----------
st.title("🎯 Multi-Domain Recommendation System (AIML Project)")

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

# ---------- Helper Function ----------
def get_recommendations(df, column, query, top_n=5):
    df = df.dropna(subset=[column])
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df[column].astype(str))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df[column].astype(str)).drop_duplicates()

    if query not in indices:
        return ["No exact match found. Try another name."]
    
    idx = indices[query]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    rec_indices = [i[0] for i in sim_scores]
    return df[column].iloc[rec_indices].tolist()

# ---------- User Selection ----------
st.sidebar.header("Choose a Recommendation Category:")
option = st.sidebar.selectbox(
    "Select type of recommendation",
    ["Movies", "Books", "Songs", "Electronics", "Foods", "Clothes"]
)

query = st.text_input(f"Enter a {option[:-1]} name to get recommendations:")

if st.button("Recommend"):
    if option == "Movies":
        st.write(get_recommendations(movies, 'title', query))
    elif option == "Books":
        st.write(get_recommendations(books, books.columns[0], query))
    elif option == "Songs":
        st.write(get_recommendations(songs, 'track_name', query))
    elif option == "Electronics":
        st.write(get_recommendations(electronics, electronics.columns[0], query))
    elif option == "Foods":
        st.write(get_recommendations(foods, foods.columns[0], query))
    elif option == "Clothes":
        st.write(get_recommendations(clothes, clothes.columns[0], query))
