from fastapi import FastAPI
from pydantic import BaseModel
from models.schemas import AnswerEvaluationRequest
from utils.text_utils import STOPWORDS
from document_storage import router as answer_sheet_router
from evaluation_router import router as evaluation_router
from teacher_exam_storage import router as teacher_exam_router
app = FastAPI(title="Automated Answer Evaluation System")

app.include_router(answer_sheet_router)
from evaluation_router import router as evaluation_router

app = FastAPI(title="Automated Answer Evaluation System")
app.include_router(teacher_exam_router)
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



