from backend.services.preprocessing import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity(model_answer: str, student_answer: str):

    model_answer = preprocess_text(model_answer)
    student_answer = preprocess_text(student_answer)

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform([model_answer, student_answer])

    similarity = cosine_similarity(vectors[0], vectors[1])

    return float(similarity[0][0])