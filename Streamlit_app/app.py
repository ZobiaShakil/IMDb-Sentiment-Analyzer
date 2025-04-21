import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import requests

# Load Sentiment Model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("imdb-distilbert-best", local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained("imdb-distilbert-best", local_files_only=True)
    return tokenizer, model

tokenizer, model = load_model()

#  Sidebar: Movie Info 
st.sidebar.markdown("## 🎥 Movie Info Lookup")
movie_title = st.sidebar.text_input("Enter movie title")
movie_plot = None

def fetch_metadata(title):
    url = f"http://www.omdbapi.com/?t={requests.utils.quote(title)}&apikey=b146a341"
    data = requests.get(url).json()
    return data if data.get("Response") == "True" else None

if movie_title:
    meta = fetch_metadata(movie_title)
    if meta:
        movie_plot = meta["Plot"]
        st.sidebar.image(meta["Poster"], width=150)
        st.sidebar.markdown(f"**{meta['Title']} ({meta['Year']})**")
        st.sidebar.markdown(f"**IMDb Rating:** {meta['imdbRating']}/10")
        st.sidebar.markdown(f"**Genre:** {meta['Genre']}")
        st.sidebar.markdown(f"**Plot Preview:** {movie_plot[:150]}…")
    else:
        st.sidebar.warning("Movie not found in OMDb.")

#  Main Section: Sentiment Analyzer 
st.title("🎬 IMDb Sentiment Analyzer")
st.write("Write a movie review and find out if it's positive or negative.")

review = st.text_area("📝 Enter your movie review here", height=200)

if st.button("🔍 Analyze Sentiment", key="analyze_btn"):
    if review.strip() == "":
        st.warning("Please enter a review before analyzing.")
    else:
        inputs = tokenizer(
            review,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        pred = int(np.argmax(probs))

        st.markdown("### Result:")
        if pred == 1:
            st.success("👍 Positive Review! 🎉✨")
            st.write("Enjoy your movie! 🍿")
            try:
                st.balloons()
            except:
                pass
        else:
            st.error("👎 Negative Review")

        st.markdown(f"**Confidence:** Positive `{probs[1]:.2f}`, Negative `{probs[0]:.2f}`")
        st.bar_chart({"Sentiment": {"Positive": probs[1], "Negative": probs[0]}})

      
