import streamlit as st
import pickle
import re

# =========================
# Load model & vectorizer
# =========================
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# =========================
# Text cleaning function
# =========================
def clean_text(text):
    text = text.lower()

    # Expand negations
    text = text.replace("don't", "do not")
    text = text.replace("doesn't", "does not")
    text = text.replace("can't", "cannot")
    text = text.replace("won't", "will not")

    # Remove special characters
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

# =========================
# Phrase rules
# =========================
strong_negative_phrases = [
    "never buy",
    "do not buy",
    "dont buy",
    "not upto the mark",
    "not up to the mark",
    "worst",
    "very bad",
    "bad product",
    "poor quality",
    "not good",
    "not satisfied",
    "waste of money"
]

strong_positive_phrases = [
    "very good",
    "excellent",
    "best product",
    "worth buying",
    "highly recommended",
    "good quality",
    "nice product",
    "value for money",
    "quality is upto the mark",
    "quality is up to the mark"
]

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Flipkart Review Sentiment Analyzer", layout="centered")

st.title("🛒 Flipkart Review Sentiment Analyzer")

review = st.text_area("Enter a product review:")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review")
    else:
        cleaned_review = clean_text(review)

        # ---- Rule-based override ----
        final_sentiment = None

        for phrase in strong_negative_phrases:
            if phrase in cleaned_review:
                final_sentiment = "negative"
                break

        if final_sentiment is None:
            for phrase in strong_positive_phrases:
                if phrase in cleaned_review and "not" not in cleaned_review:
                    final_sentiment = "positive"
                    break

        # ---- ML prediction ----
        if final_sentiment is None:
            vector = vectorizer.transform([cleaned_review])
            prediction = model.predict(vector)[0]
            final_sentiment = prediction.lower()

        # ---- Output ----
        if final_sentiment == "positive":
            st.success("✅ Predicted Sentiment: Positive 😊")
        else:
            st.error("❌ Predicted Sentiment: Negative 😞")