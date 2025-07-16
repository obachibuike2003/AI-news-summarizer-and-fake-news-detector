import streamlit as st
import joblib
import numpy as np
from transformers import pipeline

# Load summarizer from Hugging Face
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="t5-small")

# Load fake news detection model and vectorizer
@st.cache_resource
def load_classifier():
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

st.title("📰 AI News Summarizer + Fake News Detector")

# Load models
summarizer = load_summarizer()
model, vectorizer = load_classifier()

# UI input
input_text = st.text_area("Paste your news article here:", height=300)

if st.button("Analyze"):
    if not input_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        st.subheader("📄 Summary")
        summary = summarizer(input_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        st.write(summary)

        st.subheader("✅ Fake News Detection")
        X = vectorizer.transform([input_text])
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0].max()

        label = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
        st.markdown(f"**Result:** {label}")
        st.markdown(f"**Confidence:** `{proba * 100:.2f}%`")
