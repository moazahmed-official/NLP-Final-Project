"""
Lightweight preprocessing utilities for Streamlit app
"""
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure basic downloads (quiet)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_text(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    if remove_stopwords:
        tokens = [t for t in tokens if t.lower() not in STOPWORDS]
    if lemmatize:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if len(t) > 1]
    return ' '.join(tokens)


def preprocess_series(series, **kwargs):
    return series.fillna('').astype(str).apply(lambda x: preprocess_text(x, **kwargs))
