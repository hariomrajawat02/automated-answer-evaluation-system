from backend.services.keyword_extractor import extract_keywords
from backend.services.similarity import compute_similarity


STOPWORDS = {
    "is", "the", "and", "of", "to", "in", "a", "an", "for", "on", "with"
}

KEYWORD_WEIGHTS = {
    "operating": 2,
    "system": 2,
    "hardware": 2,
    "software": 2,
    "resources": 1,
    "manages": 1,
    "controls": 1
}


def evaluate_answer_logic(model_answer: str, student_answer: str):

    model = model_answer.lower()
    student = student_answer.lower()

    # Extract keywords from model answer
    keywords = extract_keywords(model_answer)

    # Find matched keywords
    matched = [word for word in keywords if word in student]
    missing = [word for word in keywords if word not in student]

    total_weight = 0
    matched_weight = 0

    for word in keywords:
        weight = KEYWORD_WEIGHTS.get(word, 1)
        total_weight += weight

        if word in student:
            matched_weight += weight

    # Keyword score (0–10 scale)
    if total_weight == 0:
        score = 0
    else:
        score = int((matched_weight / total_weight) * 10)

#feedback
    if len(matched) == len(keywords):
        feedback = "Excellent answer. You covered all key concepts."

    elif len(matched) >= len(keywords) / 2:
        feedback = (
            f"Good answer. You covered these concepts: {matched}. "
            f"But you missed: {missing}."
    )

    else:
        feedback = (
             f"The answer is partially correct. "
             f"Covered: {matched}. "
            f"Missing important concepts: {missing}."
    )
    # Semantic similarity using TF-IDF + Cosine Similarity
    similarity_score = compute_similarity(model_answer, student_answer)

    # Normalize keyword score (0–1)
    keyword_score_normalized = score / 10

    # Hybrid scoring
    final_score = (0.6 * keyword_score_normalized) + (0.4 * similarity_score)

    final_score = round(final_score * 10, 2)

    return {
        "keyword_score": score,
        "similarity_score": round(similarity_score, 2),
        "missing_keywords": missing,
        "final_score": final_score,
        "matched_keywords": matched,
        "feedback": feedback
    }