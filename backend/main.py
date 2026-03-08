from fastapi import FastAPI
from pydantic import BaseModel
from backend.services.evaluator import evaluate_answer_logic

app = FastAPI(title="Automated Answer Evaluation System")


# 1️⃣ Request data model
class AnswerEvaluationRequest(BaseModel):
    question: str
    model_answer: str
    student_answer: str


# 2️⃣ Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Backend is running successfully"
    }

# 3️⃣ Evaluation endpoint
@app.post("/evaluate")
def evaluate_answer(data: AnswerEvaluationRequest):
    return evaluate_answer_logic(
        data.model_answer,
        data.student_answer
    )