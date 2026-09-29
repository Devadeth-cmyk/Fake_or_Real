import streamlit as st
import pandas as pd
import numpy as np
import pickle
import string
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, csr_matrix
from datetime import datetime

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')

# Load the trained model and vectorizers
@st.cache_resource
def load_model_and_artifacts():
    with open('decision_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open('onehot_encoder.pkl', 'rb') as f:
        onehot_encoder = pickle.load(f)
    return model, tfidf_vectorizer, onehot_encoder

model, tfidf_vectorizer, onehot_encoder = load_model_and_artifacts()

# Preprocessing functions (from your notebook)
def remove_punc(text):
    text = str(text)
    text = text.replace("", "")
    punctuationless_text = ''.join(i for i in text if i not in string.punctuation)
    return punctuationless_text

def tokenization(text):
    word_list = nltk.word_tokenize(text)
    return word_list

stop_words_list = nltk.corpus.stopwords.words('english')
def remove_stopwords(tokens):
    stop_word_less_list = [i for i in tokens if i not in stop_words_list]
    return stop_word_less_list

lem_obj = WordNetLemmatizer()
def lemmatizing(stem_token_list):
    lem_list = [lem_obj.lemmatize(word) for word in stem_token_list]
    return lem_list

def preprocess_text(text):
    text = remove_punc(text)
    text = text.lower()
    tokens = tokenization(text)
    tokens = remove_stopwords(tokens)
    lemmatized_tokens = lemmatizing(tokens)
    return ' '.join(lemmatized_tokens)

# Streamlit App
st.title('Fake News Detection App')
st.write("Enter a news article's title, text, and subject to predict if it's real or fake.")

news_title = st.text_input('News Title')
news_text = st.text_area('News Text')
news_subject = st.selectbox('News Subject', onehot_encoder.categories_[0]) # Use categories from trained encoder

# Hardcoding date features for simplicity, as the model expects them
# For a more robust app, you might ask for a date input from the user.
current_year = datetime.now().year
current_month = datetime.now().month
current_day_of_week = datetime.now().weekday() # Monday=0, Sunday=6

if st.button('Predict'):
    if not news_text or not news_title:
        st.warning('Please provide both news title and text.')
    else:
        # Preprocess the input text
        processed_combined_text = preprocess_text(news_title + " " + news_text)

        # TF-IDF Vectorization
        text_features = tfidf_vectorizer.transform([processed_combined_text])

        # One-Hot Encoding for Subject
        subject_features = onehot_encoder.transform(pd.DataFrame([news_subject], columns=['subject']))

        # Date Features (matching the training format)
        # Note: These are simplified for the demo. In a real app, you might parse a date from the input.
        date_features = csr_matrix(np.array([[current_year, current_month, current_day_of_week]]))

        # Combine all features
        combined_features = hstack([text_features, date_features, subject_features]).tocsr()

        # Make prediction
        prediction = model.predict(combined_features)

        if prediction[0] == 0:
            st.error('Prediction: This news article is LIKELY FAKE.')
        else:
            st.success('Prediction: This news article is LIKELY REAL.')
