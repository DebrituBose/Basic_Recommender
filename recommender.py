import streamlit as st
import pandas as pd
import os

st.title("🎯 AI/ML Multi-Domain Recommender System")
st.write("Get personalized recommendations for Movies, Books, Songs, Electronics, Foods, and Clothes!")

# --- Load datasets safely ---
@st.cache_data
def load_data(filename):
    if os.path.exists(filename):
        return pd.read_csv(filename, low_memory=False)
    else:
        st.warning(f"⚠️ File not found: {filename}")
        return pd.DataFrame()

datasets = {
    "Movies": load_data("movies_small.csv"),
    "Books": load_data("books_small.csv"),
    "Songs": load_data("Spotify_small.csv"),
    "Electronics": load_data("electronics_small.csv"),
    "Foods": load_data("foods_small.csv"),
    "Clothes": load_data("clothes_small.csv")
}

# --- Choose category ---
option = st.selectbox(
    "Select a category for recommendation:",
    list(datasets.keys())
)

data = datasets[option]

if not data.empty:
    # --- Show data preview ---
    st.subheader(f"Sample {option} Data")
    st.dataframe(data.head(5))

    # --- Take user input ---
    user_input = st.text_input(
    f"Enter keywords for {option} recommendation:",
    key=f"input_{option}",
    autocomplete="off"
)


    if st.button("🔍 Get Recommendations"):
        # --- Convert columns to string and search ---
        matches = data.apply(lambda row: row.astype(str).str.contains(user_input, case=False, na=False)).any(axis=1)
        results = data[matches]

        if not results.empty:
            st.success(f"✅ Found {len(results)} recommendations!")
            st.dataframe(results.head(10))
        else:
            st.error("❌ No perfect match found. Try a different keyword.")
else:
    st.error(f"No data loaded for {option}. Please check if the CSV file exists.")


