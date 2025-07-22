import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("products.csv")
    df.fillna("", inplace=True)
    return df

# AI-based recommendation function
def recommend_products(user_input, df):
    descriptions = df['product_description'].tolist()
    descriptions.append(user_input)

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(descriptions)

    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    sim_scores = cosine_sim.flatten()

    top_indices = sim_scores.argsort()[-5:][::-1]  # top 5
    return df.iloc[top_indices]

# Streamlit App UI
st.set_page_config(page_title="Eco-Friendly Product Recommender", layout="wide")
st.title("🌿 AI-Powered Eco-Friendly Product Recommender")

df = load_data()
user_input = st.text_input("📝 Describe what you're looking for (e.g., reusable bottle, biodegradable items)")

if user_input:
    st.subheader("🔍 Recommended Products for You")
    recommended = recommend_products(user_input, df)

    for idx, row in recommended.iterrows():
        st.markdown(f"### ✅ {row['product_name']}")
        st.markdown(f"**Category**: {row['category']}")
        st.markdown(f"**Eco Score**: {row['eco_friendly_score']}")
        st.markdown(f"**Description**: {row['product_description']}")
        st.markdown("---")
