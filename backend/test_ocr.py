
import fitz
import pytesseract
from PIL import Image
import io
import os
import sys

# If Tesseract is not available through PATH,
# uncomment and update this line:
# pytesseract.pytesseract.tesseract_cmd = (
#     r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# )

def extract_text_from_pdf(pdf_path):

    if not os.path.exists(pdf_path):
        print("Error: PDF file not found!")
        return

    pdf = fitz.open(pdf_path)

    print(f"\nTotal pages: {len(pdf)}")

    all_text = []

    for page_number, page in enumerate(pdf, start=1):

        print(f"\nProcessing page {page_number}...")

        # Convert PDF page into an image
        pix = page.get_pixmap(
            dpi=220,
            colorspace=fitz.csRGB,
            alpha=False
        )

        # Convert image bytes into PIL Image
        image = Image.open(
            io.BytesIO(pix.tobytes("png"))
        )

        # Perform OCR
        extracted_text = pytesseract.image_to_string(
            image,
            lang="eng",
            config="--oem 3 --psm 6"
        )

        all_text.append(
            f"\n--- PAGE {page_number} ---\n"
            + extracted_text
        )

        print("\nExtracted Text:")
        print(extracted_text)

    pdf.close()

    # Save OCR output
    output_path = os.path.splitext(pdf_path)[0] + "_ocr_output.txt"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(all_text))

    print("\n--------------------------------")
    print("OCR completed successfully!")
    print(f"Output saved at: {output_path}")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python test_ocr.py <pdf_path>")
    else:
        extract_text_from_pdf(sys.argv[1])
