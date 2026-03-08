import re

STOPWORDS = {
    "is", "the", "and", "of", "to", "in", "a", "an", "for", "on", "with"
}

def preprocess_text(text: str):

    # lowercase
    text = text.lower()

    # remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # tokenize
    words = text.split()

    # remove stopwords
    cleaned_words = [word for word in words if word not in STOPWORDS]

    return " ".join(cleaned_words)