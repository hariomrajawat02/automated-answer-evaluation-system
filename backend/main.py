from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Automated Answer Evaluation System")


# 1️⃣ Data model FIRST
class AnswerEvaluationRequest(BaseModel):
    question: str
    model_answer: str
    student_answer: str


# 2️⃣ Health check
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Backend is running successfully"
    }

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



# 3️⃣ Evaluation logic
@app.post("/evaluate")
def evaluate_answer(data: AnswerEvaluationRequest):
    model = data.model_answer.lower()
    student = data.student_answer.lower()
    if not data.student_answer.strip():
     return {
        "score": 0,
        "matched_keywords": [],
        "feedback": "No answer provided."
     }


    keywords = {
        word for word in model.split()
        if word not in STOPWORDS
    }

    matched = [word for word in keywords if word in student]

    total_weight = 0
    matched_weight = 0

    for word in keywords:
        weight = KEYWORD_WEIGHTS.get(word, 1)
        total_weight += weight
        if word in student:
            matched_weight += weight

    if total_weight == 0:
        score = 0
    else:
        score = int((matched_weight / total_weight) * 10)

    feedback = (
        f"You covered {matched_weight} out of {total_weight} weighted concepts. "
        f"Matched keywords: {matched}"
    )

    return {
        "score": score,
        "matched_keywords": matched,
        "feedback": feedback
    }

