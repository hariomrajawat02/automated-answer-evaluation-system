def normalize_word(word: str) -> str:
    """
    Basic normalization to handle plurals and tenses
    """
    for suffix in ["ing", "es", "s"]:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word


def evaluate_answer(
    model_answer: str,
    student_answer: str
):
    STOPWORDS = {
        "is", "the", "and", "of", "to", "in", "a", "an", "for", "on", "with"
    }

    KEYWORD_WEIGHTS = {
        "operat": 2,     # operating → operat
        "system": 2,
        "hardware": 2,
        "software": 2,
        "resource": 1,  # resources → resource
        "manage": 1,    # manages → manage
        "control": 1    # controls → control
    }

    model_words = [
        normalize_word(w)
        for w in model_answer.lower().split()
    ]

    student_words = [
        normalize_word(w)
        for w in student_answer.lower().split()
    ]

    keywords = {
        word for word in model_words
        if word not in STOPWORDS
    }

    total_weight = 0
    matched_weight = 0
    matched_keywords = []

    for word in keywords:
        weight = KEYWORD_WEIGHTS.get(word, 1)
        total_weight += weight

        if word in student_words:
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
