from datetime import datetime, timezone
import os
import re

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from gridfs import GridFS
import pytesseract
from pymongo.errors import PyMongoError

from document_storage import (
    MAX_PDF_SIZE_MB,
    client,
    db,
    ocr_pdf,
    split_into_five_answers,
)

router = APIRouter(
    prefix="/teacher/exams",
    tags=["Teacher exam storage"]
)

# Separate GridFS bucket for original teacher PDFs
teacher_pdf_fs = GridFS(db, collection="teacher_exam_pdfs")

# Collection containing extracted questions, model answers, and keywords
teacher_exams = db["teacher_exams"]

# Prevent duplicate exam IDs
teacher_exams.create_index("exam_id", unique=True)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "to", "was", "were", "will", "with", "this"
}


def utc_now():
    return datetime.now(timezone.utc)


def extract_keywords_from_text(text: str) -> dict:
    """
    Extracts key words from the model answer, filters out stopwords,
    and assigns weights based on word frequency/importance.
    """
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

    keyword_counts = {}
    for word in words:
        if word not in STOPWORDS:
            keyword_counts[word] = keyword_counts.get(word, 0) + 1

    # Assign frequency-based weights (2 for key repeated terms, 1 for single occurrence)
    keyword_weights = {}
    for word, count in keyword_counts.items():
        keyword_weights[word] = 2 if count > 1 else 1

    return keyword_weights


async def read_pdf(file: UploadFile) -> bytes:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"{file.filename or 'Uploaded file'} must be a PDF."
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail=f"{file.filename} is empty."
        )

    if len(content) > MAX_PDF_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"{file.filename} exceeds the {MAX_PDF_SIZE_MB} MB limit."
        )

    return content


@router.post("/upload")
async def upload_teacher_exam(
    exam_id: str = Form(...),
    question_paper: UploadFile = File(...),
    answer_key: UploadFile = File(...)
):
    question_pdf_id = None
    answer_pdf_id = None

    try:
        exam_id = exam_id.strip()

        if not exam_id:
            raise HTTPException(
                status_code=400,
                detail="exam_id cannot be empty."
            )

        client.admin.command("ping")

        # Prevent accidental replacement of an existing exam.
        if teacher_exams.find_one({"exam_id": exam_id}):
            raise HTTPException(
                status_code=409,
                detail="This exam_id already exists."
            )

        # Read both uploaded PDFs
        question_pdf_bytes = await read_pdf(question_paper)
        answer_pdf_bytes = await read_pdf(answer_key)

        # Extract text from both PDFs
        question_pages = ocr_pdf(question_pdf_bytes)
        answer_pages = ocr_pdf(answer_pdf_bytes)

        # Split both documents into Q1-Q5 sections
        extracted_questions = split_into_five_answers(question_pages)
        extracted_answers = split_into_five_answers(answer_pages)

        # Ensure all five questions and answers were detected.
        missing_questions = [
            item["question_id"]
            for item in extracted_questions
            if item["ocr_status"] != "extracted"
        ]

        missing_answers = [
            item["question_id"]
            for item in extracted_answers
            if item["ocr_status"] != "extracted"
        ]

        if missing_questions or missing_answers:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Could not extract all five questions and answers.",
                    "missing_questions": missing_questions,
                    "missing_answers": missing_answers
                }
            )

        # Match each question with its model answer and extracted keywords.
        questions = []

        for question, answer in zip(
            extracted_questions,
            extracted_answers
        ):
            model_answer_text = answer["ocr_text"]
            generated_keywords = extract_keywords_from_text(model_answer_text)

            questions.append({
                "question_number": question["question_number"],
                "question_id": question["question_id"],
                "question": question["ocr_text"],
                "model_answer": model_answer_text,
                "keywords": generated_keywords,  # Stored permanently in DB
                "max_marks": 10,
                "question_pages": question["page_numbers"],
                "answer_pages": answer["page_numbers"]
            })

        # Store original teacher PDFs in GridFS.
        question_pdf_id = teacher_pdf_fs.put(
            question_pdf_bytes,
            filename=question_paper.filename,
            contentType="application/pdf",
            metadata={
                "exam_id": exam_id,
                "document_type": "question_paper"
            }
        )

        answer_pdf_id = teacher_pdf_fs.put(
            answer_pdf_bytes,
            filename=answer_key.filename,
            contentType="application/pdf",
            metadata={
                "exam_id": exam_id,
                "document_type": "answer_key"
            }
        )

        now = utc_now()

        exam_record = {
            "exam_id": exam_id,
            "question_paper": {
                "filename": question_paper.filename,
                "gridfs_id": question_pdf_id
            },
            "answer_key": {
                "filename": answer_key.filename,
                "gridfs_id": answer_pdf_id
            },
            "questions": questions,
            "question_count": 5,
            "total_max_marks": 50,
            "created_at": now,
            "status": "ready"
        }

        result = teacher_exams.insert_one(exam_record)

        return {
            "status": "ok",
            "message": "Teacher exam uploaded, processed, and keywords generated successfully.",
            "exam_id": exam_id,
            "teacher_exam_id": str(result.inserted_id),
            "question_count": 5,
            "total_max_marks": 50,
            "questions": [
                {
                    "question_id": item["question_id"],
                    "question": item["question"],
                    "model_answer": item["model_answer"],
                    "keywords": item["keywords"],
                    "max_marks": item["max_marks"]
                }
                for item in questions
            ]
        }

    except HTTPException:
        raise

    except (ValueError, PyMongoError, pytesseract.TesseractNotFoundError) as exc:
        if question_pdf_id is not None:
            teacher_pdf_fs.delete(question_pdf_id)

        if answer_pdf_id is not None:
            teacher_pdf_fs.delete(answer_pdf_id)

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        ) from exc

    except Exception as exc:
        if question_pdf_id is not None:
            teacher_pdf_fs.delete(question_pdf_id)

        if answer_pdf_id is not None:
            teacher_pdf_fs.delete(answer_pdf_id)

        raise HTTPException(
            status_code=500,
            detail=f"Teacher exam processing failed: {type(exc).__name__}"
        ) from exc

    finally:
        await question_paper.close()
        await answer_key.close()


@router.get("/{exam_id}")
def get_teacher_exam(exam_id: str):
    exam = teacher_exams.find_one({"exam_id": exam_id})

    if not exam:
        raise HTTPException(
            status_code=404,
            detail="Teacher exam not found."
        )

    return {
        "exam_id": exam["exam_id"],
        "question_count": exam["question_count"],
        "total_max_marks": exam["total_max_marks"],
        "status": exam["status"],
        "questions": exam["questions"],
        "created_at": exam["created_at"].isoformat()
    }