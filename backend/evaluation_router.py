
from datetime import datetime, timezone
import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from embedding_model import generate_embedding
from cosine_similarity import compute_cosine_similarity

from document_storage import db, documents, answers


router = APIRouter(prefix="", tags=["Evaluation"])


# -------------------- MongoDB Collections --------------------

teacher_exams = db["teacher_exams"]

evaluation_results = db["evaluation_results"]
evaluation_progress = db["evaluation_progress"]


# One result per student per exam.
evaluation_results.create_index(
    [("exam_id", 1), ("student_id", 1)],
    unique=True
)

# One progress record per exam.
evaluation_progress.create_index(
    [("exam_id", 1)],
    unique=True
)


def utc_now():
    return datetime.now(timezone.utc)


# -------------------- Request Models --------------------

class AnswerEvaluationRequest(BaseModel):
    question: str
    model_answer: str
    student_answer: str


class BatchEvaluationRequest(BaseModel):
    exam_id: str
    start_student_id: str | None = None
    end_student_id: str


# -------------------- Existing Scoring Configuration --------------------

STOPWORDS = {
    "is", "the", "and", "of", "to", "in",
    "a", "an", "for", "on", "with"
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


# -------------------- Single Answer Evaluation --------------------

def evaluate_single_answer(
    question: str,
    model_answer: str,
    student_answer: str
):
    """
    Evaluate one student answer using the existing scoring algorithm.

    Maximum score: 10
    Semantic similarity: 60%
    Keyword score: 30%
    Concept coverage: 10%
    """

    if not student_answer or not student_answer.strip():
        return {
            "score": 0,
            "similarity": 0,
            "embedding_score": 0,
            "keyword_score": 0,
            "concept_coverage": 0,
            "matched_keywords": [],
            "feedback": "No answer provided."
        }

    model = model_answer.lower()
    student = student_answer.lower()

    model_vector = generate_embedding(model)
    student_vector = generate_embedding(student)

    similarity = compute_cosine_similarity(
        model_vector,
        student_vector
    )

    embedding_score = round(similarity * 10)

    keywords = {
        word for word in model.split()
        if word not in STOPWORDS
    }

    matched = sorted(
        word for word in keywords
        if word in student
    )

    concept_coverage = (
        len(matched) / len(keywords)
        if keywords else 0
    )

    total_weight = 0
    matched_weight = 0

    for word in keywords:

        weight = KEYWORD_WEIGHTS.get(word, 1)

        total_weight += weight

        if word in student:
            matched_weight += weight

    if total_weight == 0:
        keyword_score = 0
    else:
        keyword_score = int(
            (matched_weight / total_weight) * 10
        )

    final_score = int(
        (embedding_score * 0.6)
        + (keyword_score * 0.3)
        + (concept_coverage * 10 * 0.1)
    )

    return {
        "score": final_score,
        "similarity": similarity,
        "embedding_score": embedding_score,
        "keyword_score": keyword_score,
        "concept_coverage": concept_coverage,
        "matched_keywords": matched,
        "feedback": f"Semantic similarity score: {similarity:.2f}"
    }


# -------------------- Existing Single Evaluation Endpoint --------------------

@router.post("/evaluate")
def evaluate_answer(data: AnswerEvaluationRequest):

    return evaluate_single_answer(
        data.question,
        data.model_answer,
        data.student_answer
    )


# -------------------- Student ID Utilities --------------------

def parse_student_id(student_id: str):

    """
    Supports IDs such as STU001, STU002, STU050.

    Returns:
        prefix, numeric_part, number_of_digits
    """

    match = re.fullmatch(r"([A-Za-z_-]*)(\d+)", student_id)

    if not match:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid student ID format: {student_id}. "
                   "Expected format such as STU001."
        )

    prefix = match.group(1)
    number_text = match.group(2)

    return prefix, int(number_text), len(number_text)


def format_student_id(prefix: str, number: int, width: int):

    return f"{prefix}{number:0{width}d}"


# -------------------- Batch Evaluation Endpoint --------------------

@router.post("/evaluate/batch")
def evaluate_student_range(data: BatchEvaluationRequest):

    """
    Evaluate student answer sheets for a given exam and student ID range.

    If start_student_id is omitted, evaluation resumes from the
    last successfully evaluated student ID + 1.

    Student ID range is inclusive.
    """

    # 1. Find teacher's exam.
    teacher_exam = teacher_exams.find_one({
        "exam_id": data.exam_id
    })

    if not teacher_exam:
        raise HTTPException(
            status_code=404,
            detail=f"Exam {data.exam_id} not found in teacher records."
        )

    teacher_questions = teacher_exam.get("questions", [])

    if not teacher_questions:
        raise HTTPException(
            status_code=400,
            detail="No teacher questions or answer key found for this exam."
        )

    # 2. Parse ending student ID.
    end_prefix, end_number, end_width = parse_student_id(
        data.end_student_id
    )

    # 3. Determine starting student ID.
    if data.start_student_id:

        start_prefix, start_number, start_width = parse_student_id(
            data.start_student_id
        )

        if start_prefix != end_prefix:
            raise HTTPException(
                status_code=400,
                detail="Starting and ending student IDs must have the same prefix."
            )

        id_width = max(start_width, end_width)

    else:

        # Resume from the last computed student.
        progress = evaluation_progress.find_one({
            "exam_id": data.exam_id
        })

        if not progress:
            raise HTTPException(
                status_code=400,
                detail="No previous evaluation progress found. "
                       "Provide start_student_id for the first batch."
            )

        if progress["id_prefix"] != end_prefix:
            raise HTTPException(
                status_code=400,
                detail="Student ID prefix does not match saved progress."
            )

        start_number = progress["last_student_number"] + 1

        id_width = max(
            progress["id_width"],
            end_width
        )

        start_prefix = end_prefix

    if start_number > end_number:
        raise HTTPException(
            status_code=400,
            detail="Starting student ID is greater than ending student ID. "
                   "No students remain in this range."
        )

    # 4. Retrieve the latest uploaded answer sheet per student.
    student_pattern = f"^{re.escape(start_prefix)}\\d+$"

    student_documents = documents.find({
        "exam_id": data.exam_id,
        "student_id": {
            "$regex": student_pattern
        }
    }).sort("uploaded_at", -1)

    latest_documents = {}

    for doc in student_documents:

        student_id = doc["student_id"]

        prefix, student_number, _ = parse_student_id(student_id)

        if prefix != start_prefix:
            continue

        if not (start_number <= student_number <= end_number):
            continue

        # Documents are sorted newest first.
        # Keep only the latest upload for each student.
        if student_id not in latest_documents:
            latest_documents[student_id] = doc

    # Sort students numerically, not alphabetically.
    selected_students = sorted(
        latest_documents.values(),
        key=lambda doc: parse_student_id(doc["student_id"])[1]
    )

    if not selected_students:
        raise HTTPException(
            status_code=404,
            detail="No uploaded student answer sheets found in the requested range."
        )

    # 5. Evaluate each student.
    processed_results = []
    last_processed_student_id = None

    for student_doc in selected_students:

        student_id = student_doc["student_id"]

        _, student_number, student_width = parse_student_id(
            student_id
        )

        # Retrieve the student's five extracted answers.
        student_answers = list(
            answers.find({
                "document_id": student_doc["_id"]
            })
        )

        answer_map = {
            item["question_number"]: item["ocr_text"]
            for item in student_answers
        }

        question_results = []
        total_marks = 0
        total_max_marks = 0

        # Evaluate each teacher question against its student answer.
        for teacher_question in sorted(
            teacher_questions,
            key=lambda item: item["question_number"]
        ):

            question_number = teacher_question["question_number"]

            question_text = teacher_question["question"]
            model_answer = teacher_question["model_answer"]

            max_marks = teacher_question.get("max_marks", 10)

            student_answer = answer_map.get(
                question_number,
                ""
            )

            evaluation = evaluate_single_answer(
                question_text,
                model_answer,
                student_answer
            )

            # Convert score out of 10 into marks.
            marks_awarded = round(
                (evaluation["score"] / 10) * max_marks,
                2
            )

            total_marks += marks_awarded
            total_max_marks += max_marks

            question_results.append({
                "question_number": question_number,
                "question_id": f"Q{question_number}",
                "question": question_text,
                "student_answer": student_answer,
                "model_answer": model_answer,
                "max_marks": max_marks,
                "marks_awarded": marks_awarded,
                "evaluation": evaluation
            })

        total_marks = round(total_marks, 2)

        # 6. Prepare the student's complete evaluation record.
        result_document = {
            "exam_id": data.exam_id,
            "student_id": student_id,
            "source_document_id": student_doc["_id"],
            "questions": question_results,
            "total_score": total_marks,
            "total_max_marks": total_max_marks,
            "percentage": round(
                (total_marks / total_max_marks) * 100, 2
            ) if total_max_marks else 0,
            "evaluated_at": utc_now(),
            "status": "completed"
        }

        # 7. Save result. Re-evaluation updates the existing record.
        evaluation_results.update_one(
            {
                "exam_id": data.exam_id,
                "student_id": student_id
            },
            {
                "$set": result_document,
                "$setOnInsert": {
                    "created_at": utc_now()
                }
            },
            upsert=True
        )

        # 8. Update progress only after the student's result is saved.
        current_progress = evaluation_progress.find_one({
            "exam_id": data.exam_id
        })

        if (
            current_progress is None
            or student_number >= current_progress["last_student_number"]
        ):

            evaluation_progress.update_one(
                {
                    "exam_id": data.exam_id
                },
                {
                    "$set": {
                        "exam_id": data.exam_id,
                        "last_student_id": student_id,
                        "last_student_number": student_number,
                        "id_prefix": start_prefix,
                        "id_width": max(id_width, student_width),
                        "updated_at": utc_now()
                    }
                },
                upsert=True
            )

        last_processed_student_id = student_id

        processed_results.append({
            "student_id": student_id,
            "total_score": total_marks,
            "total_max_marks": total_max_marks,
            "percentage": result_document["percentage"],
            "status": "completed"
        })

    # 9. Get the latest saved progress.
    final_progress = evaluation_progress.find_one({
        "exam_id": data.exam_id
    })

    return {
        "status": "completed",
        "message": "Batch evaluation completed successfully.",
        "exam_id": data.exam_id,
        "requested_start_student_id": format_student_id(
            start_prefix,
            start_number,
            id_width
        ),
        "requested_end_student_id": data.end_student_id,
        "students_processed": len(processed_results),
        "last_processed_in_this_batch": last_processed_student_id,
        "last_computed_student_id": final_progress["last_student_id"],
        "results": processed_results
    }
