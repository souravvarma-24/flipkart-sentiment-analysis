# Sentiment Analysis of Flipkart Product Reviews

This project performs sentiment analysis on Flipkart product reviews using machine learning and natural language processing (NLP).  
The system classifies user reviews as **Positive** or **Negative** and provides real-time predictions through a web application.

---

## 🎯 Objective

- Classify Flipkart product reviews into **Positive** or **Negative**
- Understand customer sentiment from textual reviews
- Build a real-time sentiment prediction web application

---

## 📊 Dataset

- Dataset contains **8,518 Flipkart product reviews**
- Product: **YONEX MAVIS 350 Nylon Shuttle**
- Dataset was provided as part of the project
- **No manual web scraping was performed**

### Dataset Features:
- Reviewer Name  
- Reviewer Rating  
- Review Title  
- Review Text  
- Place of Review  
- Date of Review  
- Up Votes  
- Down Votes  

---

## 🧹 Data Preprocessing

- Converted text to lowercase
- Removed special characters and punctuation
- Removed stopwords
- Applied lemmatization
- Handled common negation phrases (e.g., *not good*, *never buy*)

---

## 🔍 Feature Extraction Techniques

The following text vectorization techniques were implemented:

- Bag of Words (BoW)
- TF-IDF (Term Frequency–Inverse Document Frequency)
- Word2Vec
- BERT (Sentence Transformers)

---

## 🤖 Models Used

- TF-IDF + Logistic Regression
- Bag of Words + Logistic Regression
- Naive Bayes
- Word2Vec + Logistic Regression
- BERT Embeddings + Logistic Regression

---

## 📈 Model Evaluation

- **Evaluation Metric:** F1-Score
- Models were trained and evaluated using the test dataset
- Best-performing model was selected for deployment

---

## 🌐 Web Application

- Developed using **Streamlit**
- Accepts real-time user review input
- Predicts sentiment as:
  - ✅ Positive
  - ❌ Negative
- Simple and user-friendly interface

---

## 🚀 Deployment

- Streamlit-based web application
- Designed for deployment on **AWS EC2**
- Supports real-time sentiment prediction

---

## 🛠 Technologies Used

- Python
- Pandas, NumPy
- NLTK
- Scikit-learn
- Gensim
- Sentence Transformers
- Streamlit

---

## 📁 Project Structure

```text
├── app.py
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── Untitled62.ipynb
├── README.md

---


## ✅ Conclusion

This project successfully demonstrates an end-to-end **Sentiment Analysis system for Flipkart product reviews**.  
Multiple NLP techniques such as **Bag of Words, TF-IDF, Word2Vec, and BERT embeddings** were implemented and evaluated using **F1-Score** to ensure reliable performance.

A **Streamlit-based web application** was developed to provide **real-time sentiment prediction** for user-entered reviews.  
The project follows proper data preprocessing, model training, evaluation, and deployment practices, making it suitable for real-world applications and scalable deployment on **AWS EC2**.

Overall, this project provides hands-on experience in **Natural Language Processing, Machine Learning, and Model Deployment**, aligning fully with the project requirements.
