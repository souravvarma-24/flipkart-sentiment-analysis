import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------
# NLTK setup
# -------------------------
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -------------------------
# Load trained ML model
# -------------------------
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# -------------------------
# OPINION VERBS (CRITICAL)
# -------------------------
NEGATIVE_OPINION_VERBS = {
    "disagree", "regret", "complain", "complaint", "hate",
    "avoid", "refuse", "return", "refund", "replace",
    "disappointed", "unsatisfied", "dissatisfied"
}

# -------------------------
# Word lexicons
# -------------------------
POSITIVE_WORDS = {
    "good", "nice", "excellent", "amazing", "awesome",
    "perfect", "happy", "satisfied", "best", "great",
    "worth", "value", "quality", "recommended",
    "comfortable", "useful", "reliable", "suitable"
}

NEGATIVE_WORDS = {
    "bad", "worst", "poor", "waste", "broken",
    "damage", "cheap", "fake", "defective",
    "problem", "issue", "late", "delay",
    "useless", "uncomfortable", "unsuitable"
}

# -------------------------
# Cleaning
# -------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return words

# -------------------------
# Generic negation detector
# -------------------------
def has_negation(text):
    return bool(re.search(r"\bnot\s+\w+", text))

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(
    page_title="Flipkart Product Review Sentiment Analysis",
    layout="centered"
)

st.title("Flipkart Product Review Sentiment Analysis")
st.write("Enter a product review to predict sentiment")

review = st.text_area("Review Text")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review")
    else:
        text = review.lower()
        tokens = clean_text(review)

        # ========= 1️⃣ OPINION VERBS =========
        if any(v in tokens for v in NEGATIVE_OPINION_VERBS):
            st.error("Negative Review 😞")
            st.stop()

        # ========= 2️⃣ NEGATION =========
        if has_negation(text):
            st.error("Negative Review 😞")
            st.stop()

        # ========= 3️⃣ TOKEN SCORING =========
        pos_score = sum(1 for w in tokens if w in POSITIVE_WORDS)
        neg_score = sum(1 for w in tokens if w in NEGATIVE_WORDS)

        if pos_score > 0 and neg_score > 0:
            st.info("Mixed Review 😐")
            st.stop()

        if neg_score > pos_score:
            st.error("Negative Review 😞")
            st.stop()

        if pos_score > neg_score:
            st.success("Positive Review 😊")
            st.stop()

        # ========= 4️⃣ ML FALLBACK =========
        cleaned = " ".join(tokens)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction.lower() == "positive":
            st.success("Positive Review 😊")
        else:
            st.error("Negative Review 😞")
