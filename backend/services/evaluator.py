def evaluate_answer(
    model_answer: str,
    student_answer: str
):
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

    model = model_answer.lower()
    student = student_answer.lower()

    keywords = {
        word for word in model.split()
        if word not in STOPWORDS
    }

    total_weight = 0
    matched_weight = 0
    matched_keywords = []

    for word in keywords:
        weight = KEYWORD_WEIGHTS.get(word, 1)
        total_weight += weight

        if word in student:
            matched_weight += weight
            matched_keywords.append(word)

    score = (
        int((matched_weight / total_weight) * 10)
        if total_weight > 0 else 0
    )

    feedback = (
        f"You covered {matched_weight} out of "
        f"{total_weight} weighted concepts."
    )

    return {
        "score": score,
        "matched_keywords": matched_keywords,
        "feedback": feedback
    }
