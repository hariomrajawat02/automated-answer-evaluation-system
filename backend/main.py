from fastapi import FastAPI
from pydantic import BaseModel
from evaluation_router import router as evaluation_router

app = FastAPI(title="Automated Answer Evaluation System")
app.include_router(evaluation_router)

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



