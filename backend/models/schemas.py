from pydantic import BaseModel

class AnswerEvaluationRequest(BaseModel):
    question: str
    model_answer: str
    student_answer: str
