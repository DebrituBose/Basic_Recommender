import streamlit as st
import pandas as pd
from recommender import recommend_item

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Recommender",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
        }
        .main {
            background-color: #ffffff;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2C3E50;
            text-align: center;
            font-family: 'Poppins', sans-serif;
        }
        .stButton>button {
            background-color: #4B9CD3;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border-radius: 10px;
            border: none;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #1E88E5;
            transform: scale(1.05);
        }
        .recommend-card {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("<h1>🎬 Smart Recommender System</h1>", unsafe_allow_html=True)
st.write("Discover personalized recommendations for movies, books, songs, and more!")

# --- Category Selection ---
category = st.selectbox("Choose a category", ["Movies", "Books", "Songs", "Electronics", "Foods", "Clothes"])

# --- Input Box ---
keyword = st.text_input(f"🔍 Enter keywords for {category} recommendation:")

# --- Recommend Button ---
if st.button("✨ Get Recommendations"):
    if keyword.strip():
        results = recommend_item(category.lower(), keyword)
        if len(results) > 0:
            st.subheader(f"📦 Recommended {category}:")
            for r in results:
                st.markdown(f"<div class='recommend-card'>✅ {r}</div>", unsafe_allow_html=True)
        else:
            st.error("😕 No perfect match found. Try different keywords.")
    else:
        st.warning("Please enter a keyword before searching!")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
