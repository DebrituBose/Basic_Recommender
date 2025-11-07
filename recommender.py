import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------
# Step 1: Load all datasets
# --------------------------
@st.cache_data
def load_data():
    books = pd.read_csv("books_small.csv", low_memory=False)
    movies = pd.read_csv("movies_small.csv", low_memory=False)
    songs = pd.read_csv("songs_small.csv", low_memory=False)
    electronics = pd.read_csv("electronics_small.csv", low_memory=False)
    foods = pd.read_csv("foods_small.csv", low_memory=False)
    clothes = pd.read_csv("clothes_small.csv", low_memory=False)
    return books, movies, songs, electronics, foods, clothes

books, movies, songs, electronics, foods, clothes = load_data()

# --------------------------
# Step 2: Create TF-IDF helper
# --------------------------
def build_similarity_matrix(df, text_column):
    df[text_column] = df[text_column].fillna('')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df[text_column])
    similarity = cosine_similarity(tfidf_matrix)
    return similarity

# --------------------------
# Step 3: General recommend function
# --------------------------
def get_recommendations(user_input, df, text_column, name):
    titles = df[text_column].dropna().tolist()
    close_matches = difflib.get_close_matches(user_input, titles, n=1, cutoff=0.2)
    
    if not close_matches:
        st.warning(f"No close matches found in {name}. Try a different word.")
        return
    
    matched_title = close_matches[0]
    st.success(f"🔍 Showing {name} similar to: {matched_title}")
    
    try:
        similarity = build_similarity_matrix(df, text_column)
        index = df[df[text_column] == matched_title].index[0]
        scores = list(enumerate(similarity[index]))
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
        for i, score in sorted_scores:
            st.write(f"👉 {df[text_column].iloc[i]}")
    except Exception as e:
        st.warning(f"Couldn’t generate recommendations due to: {e}")

# --------------------------
# Step 4: Streamlit UI
# --------------------------
st.title("🎯 Multi-Domain AI Recommender System")

option = st.selectbox(
    "Choose what you want recommendations for:",
    ["Books", "Movies", "Songs", "Electronics", "Foods", "Clothes"]
)

user_input = st.text_input("Enter a title, name, or keyword:")

if st.button("Recommend"):
    if option == "Books":
        get_recommendations(user_input, books, 'title', 'Books')
    elif option == "Movies":
        get_recommendations(user_input, movies, 'title', 'Movies')
    elif option == "Songs":
        get_recommendations(user_input, songs, 'track_name', 'Songs')
    elif option == "Electronics":
        get_recommendations(user_input, electronics, 'title', 'Electronics')
    elif option == "Foods":
        get_recommendations(user_input, foods, 'Item_Name', 'Foods')
    elif option == "Clothes":
        get_recommendations(user_input, clothes, 'Item_Name', 'Clothes')
