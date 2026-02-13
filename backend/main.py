from fastapi import FastAPI
from pydantic import BaseModel
from models.schemas import AnswerEvaluationRequest
from utils.text_utils import STOPWORDS

app = FastAPI(title="Automated Answer Evaluation System")

# 2️⃣ Health check
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Backend is running successfully"
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

