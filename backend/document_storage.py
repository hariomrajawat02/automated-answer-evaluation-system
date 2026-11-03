"""
MongoDB-backed storage for scanned student answer sheets.

Designed as an add-on module for the existing FastAPI backend.
- Stores the original PDF in MongoDB GridFS.
- OCRs each page with Tesseract.
- Splits OCR text into Q1..Q5 answer records.
- Stores one document record and five answer records in MongoDB.

Expected answer headings in the scanned sheet include:
    Q1 / Q.1 / Question 1
    Q2 / Q.2 / Question 2
    ...
This is rule-based segmentation. If handwriting/OCR does not preserve question
headings, review or improve the segmentation logic before using the text for grading.
"""

import os
import re
import hashlib
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

from dotenv import load_dotenv 
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from bson import ObjectId
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from gridfs import GridFS
from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError

load_dotenv()
# -------------------- Configuration --------------------

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "answer_evaluation_system")
MAX_PDF_SIZE_MB = int(os.getenv("MAX_PDF_SIZE_MB", "25"))
OCR_DPI = int(os.getenv("OCR_DPI", "220"))

# If Tesseract is not on PATH on Windows, set TESSERACT_CMD in .env, e.g.:
# TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
_tesseract_cmd = os.getenv("TESSERACT_CMD")
if _tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd

client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
db = client[MONGODB_DATABASE]
pdf_fs = GridFS(db, collection="answer_sheet_pdfs")
documents = db["answer_sheet_documents"]
answers = db["extracted_answers"]

# Useful indexes for retrieving a student's answers and preventing accidental
# duplicate answer records for the same question in one document.
documents.create_index([("uploaded_at", ASCENDING)])
answers.create_index(
    [("document_id", ASCENDING), ("question_number", ASCENDING)],
    unique=True,
)

router = APIRouter(prefix="/answer-sheets", tags=["Answer sheet storage"])


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ocr_pdf(pdf_bytes: bytes) -> list[dict[str, Any]]:
    """Return OCR text and page metadata for each PDF page."""
    page_results = []

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
            if pdf.page_count == 0:
                raise ValueError("The PDF contains no pages.")

            for page_index, page in enumerate(pdf):
                # Use embedded text when available, but OCR rendered page images
                # when the PDF is scanned or has no selectable text.
                text = page.get_text("text").strip()
                if not text:
                    pix = page.get_pixmap(
                        matrix=fitz.Matrix(OCR_DPI / 72, OCR_DPI / 72),
                        alpha=False,
                    )
                    image = Image.open(BytesIO(pix.tobytes("png")))
                    text = pytesseract.image_to_string(image, config="--psm 6").strip()

                page_results.append({
                    "page_number": page_index + 1,
                    "text": text,
                })
    except (fitz.FileDataError, RuntimeError) as exc:
        raise ValueError(f"Could not read the PDF: {exc}") from exc

    return page_results


def split_into_five_answers(page_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Split combined OCR text by explicit Q1..Q5 / Question 1..5 headings.

    Text before Q1 is not assigned to an answer. Missing questions are returned
    with empty text so the database always contains five answer records.
    """
    full_text = "\n".join(
        f"\n[[PAGE:{p['page_number']}]]\n{p['text']}" for p in page_results
    )

    # Recognizes Q1, Q.1, Question 1, and optional punctuation at line starts.
    heading = re.compile(
        r"(?im)^[ \t]*(?:question[ \t]*|q[ \t]*\.?[ \t]*)"
        r"([1-5])(?:[ \t]*[.):\-])?[ \t]*"
    )
    matches = list(heading.finditer(full_text))
    extracted: dict[int, dict[str, Any]] = {}

    for index, match in enumerate(matches):
        question_number = int(match.group(1))
        # If a heading is repeated (e.g. OCR repeated it), retain the first
        # non-empty segment for that question.
        end = matches[index + 1].start() if index + 1 < len(matches) else len(full_text)
        segment = full_text[match.end():end].strip()
        segment = re.sub(r"\[\[PAGE:(\d+)\]\]", "", segment).strip()
        page_numbers = sorted({
            int(n) for n in re.findall(r"\[\[PAGE:(\d+)\]\]", full_text[match.start():end])
        })
        if question_number not in extracted or (
            not extracted[question_number]["ocr_text"] and segment
        ):
            extracted[question_number] = {
                "question_number": question_number,
                "question_id": f"Q{question_number}",
                "ocr_text": segment,
                "page_numbers": page_numbers,
                "ocr_status": "extracted" if segment else "empty",
            }

    # Always return exactly five records, one for each expected question.
    result = []
    for number in range(1, 6):
        result.append(extracted.get(number, {
            "question_number": number,
            "question_id": f"Q{number}",
            "ocr_text": "",
            "page_numbers": [],
            "ocr_status": "not_found",
        }))
    return result


@router.post("/upload")
async def upload_answer_sheet(
    file: UploadFile = File(...),
    student_id: str = Form(...),
    exam_id: str = Form(...),
):
    """
    Upload one scanned PDF.

    multipart/form-data fields:
      - file: scanned PDF
      - student_id: student/roll identifier
      - exam_id: identifier for the exam
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    pdf_bytes = await file.read()
    max_bytes = MAX_PDF_SIZE_MB * 1024 * 1024
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")
    if len(pdf_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"PDF exceeds the {MAX_PDF_SIZE_MB} MB upload limit.",
        )

    try:
        client.admin.command("ping")
        page_results = ocr_pdf(pdf_bytes)
        answer_records = split_into_five_answers(page_results)

        # Save the source PDF in GridFS rather than placing a large binary blob
        # directly inside a normal MongoDB document.
        pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
        gridfs_id = pdf_fs.put(
            pdf_bytes,
            filename=file.filename,
            contentType="application/pdf",
            metadata={
                "student_id": student_id,
                "exam_id": exam_id,
                "sha256": pdf_hash,
            },
        )

        now = utc_now()
        document_record = {
            "student_id": student_id,
            "exam_id": exam_id,
            "original_filename": file.filename,
            "pdf_gridfs_id": gridfs_id,
            "pdf_sha256": pdf_hash,
            "page_count": len(page_results),
            "uploaded_at": now,
            "ocr_completed_at": now,
            "ocr_status": "completed",
            "question_count_expected": 5,
            "question_count_detected": sum(
                1 for answer in answer_records if answer["ocr_status"] == "extracted"
            ),
        }
        inserted_document = documents.insert_one(document_record)
        document_id = inserted_document.inserted_id

        answer_documents = []
        for answer in answer_records:
            answer_documents.append({
                **answer,
                "document_id": document_id,
                "student_id": student_id,
                "exam_id": exam_id,
                "created_at": now,
            })

        if answer_documents:
            answers.insert_many(answer_documents, ordered=True)

        return {
            "status": "ok",
            "message": "PDF stored and OCR processing completed.",
            "document_id": str(document_id),
            "pdf_gridfs_id": str(gridfs_id),
            "student_id": student_id,
            "exam_id": exam_id,
            "page_count": len(page_results),
            "expected_answers": 5,
            "detected_answers": document_record["question_count_detected"],
            "answers": [
                {
                    "question_id": item["question_id"],
                    "ocr_status": item["ocr_status"],
                    "page_numbers": item["page_numbers"],
                    "ocr_text": item["ocr_text"],
                }
                for item in answer_records
            ],
        }
    except HTTPException:
        raise
    except (ValueError, PyMongoError, pytesseract.TesseractNotFoundError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        # Avoid returning internal stack traces to the client.
        raise HTTPException(
            status_code=500,
            detail=f"Answer-sheet processing failed: {type(exc).__name__}",
        ) from exc
    finally:
        await file.close()


@router.get("/{document_id}")
def get_answer_sheet(document_id: str):
    """Fetch document metadata and its five extracted answer records."""
    if not ObjectId.is_valid(document_id):
        raise HTTPException(status_code=400, detail="Invalid document_id.")

    doc = documents.find_one({"_id": ObjectId(document_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Answer sheet not found.")

    answer_docs = list(
        answers.find({"document_id": doc["_id"]}).sort("question_number", ASCENDING)
    )

    return {
        "document_id": str(doc["_id"]),
        "student_id": doc["student_id"],
        "exam_id": doc["exam_id"],
        "original_filename": doc["original_filename"],
        "pdf_gridfs_id": str(doc["pdf_gridfs_id"]),
        "page_count": doc["page_count"],
        "uploaded_at": doc["uploaded_at"].isoformat(),
        "ocr_status": doc["ocr_status"],
        "answers": [
            {
                "question_id": item["question_id"],
                "question_number": item["question_number"],
                "ocr_text": item["ocr_text"],
                "page_numbers": item["page_numbers"],
                "ocr_status": item["ocr_status"],
            }
            for item in answer_docs
        ],
    }


@router.get("/{document_id}/pdf")
def download_original_pdf(document_id: str):
    """Return metadata for the stored original PDF; use GridFS to stream in production."""
    if not ObjectId.is_valid(document_id):
        raise HTTPException(status_code=400, detail="Invalid document_id.")

    doc = documents.find_one({"_id": ObjectId(document_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Answer sheet not found.")

    # GridFS file content can be streamed using StreamingResponse if needed.
    grid_file = pdf_fs.get(doc["pdf_gridfs_id"])
    from fastapi.responses import Response

    return Response(
        content=grid_file.read(),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{doc["original_filename"]}"'
        },
    )
