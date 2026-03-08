import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words("english"))

def extract_keywords(text: str):

    tokens = word_tokenize(text.lower())

    keywords = [
        word for word in tokens
        if word.isalpha() and word not in STOPWORDS
    ]

    return list(set(keywords))