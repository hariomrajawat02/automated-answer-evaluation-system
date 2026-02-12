from fastapi import FastAPI
from pydantic import BaseModel
from services.evaluator import evaluate_answer

app = FastAPI(title="Automated Answer Evaluation System")
class AnswerEvaluationRequest(BaseModel):
    question: str
    model_answer: str
    student_answer: str
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Backend is running successfully"
    }
@app.post("/evaluate")
def evaluate_api(data: AnswerEvaluationRequest):
    result = evaluate_answer(
        data.model_answer,
        data.student_answer
    )
    return result
