from pathlib import Path

import pandas as pd
import pytesseract
from PIL import Image
from docx import Document
import pymupdf


def load_txt(file_path: Path):
    text = file_path.read_text(encoding="utf-8")

    return [
        {
            "text": text,
            "metadata": {
                "source": file_path.name,
                "file_type": "txt",
                "page":None,
                "section":None
            },
        }
    ]


def load_pdf(file_path: Path):
    documents = []

    # A file named .pdf must contain the PDF signature, not merely text with a
    # PDF extension. Checking it here gives a clear remediation path.
    with file_path.open("rb") as file:
        pdf_signature = file.read(5)

    if pdf_signature != b"%PDF-":
        raise ValueError(
            "File has a .pdf extension but is not a valid PDF. "
            "Convert it to PDF or rename it with a .txt extension."
        )

    pdf = pymupdf.open(file_path)

    for page_number, page in enumerate(pdf, start=1):
        text = page.get_text().strip()

        # If the PDF is scanned, use OCR
        if not text:
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
            image = Image.frombytes(
                "RGB",
                [pixmap.width, pixmap.height],
                pixmap.samples,
            )
            text = pytesseract.image_to_string(image)

        if text.strip():
            documents.append(
                {
                    "text": text,
                    "metadata": {
                        "source": file_path.name,
                        "page": page_number,
                        "file_type": "pdf",
                    },
                }
            )

    return documents


def load_docx(file_path: Path):
    document = Document(file_path)

    text_parts = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if text:
            text_parts.append(text)

    # Extract tables
    for table in document.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells)
            text_parts.append(row_text)

    text = "\n".join(text_parts)

    return [
        {
            "text": text,
            "metadata": {
                "source": file_path.name,
                "file_type": "docx",
            },
        }
    ]


def load_image(file_path: Path):
    image = Image.open(file_path)

    text = pytesseract.image_to_string(image)

    return [
        {
            "text": text,
            "metadata": {
                "source": file_path.name,
                "file_type": "image",
            },
        }
    ]


def load_csv_or_excel(file_path: Path):
    if file_path.suffix.lower() == ".csv":
        dataframe = pd.read_csv(file_path)
    else:
        dataframe = pd.read_excel(file_path)

    text_rows = []

    for index, row in dataframe.iterrows():
        row_text = " | ".join(
            f"{column}: {row[column]}"
            for column in dataframe.columns
        )

        text_rows.append(row_text)

    return [
        {
            "text": "\n".join(text_rows),
            "metadata": {
                "source": file_path.name,
                "file_type": file_path.suffix.lower(),
            },
        }
    ]


def load_document(file_path):
    file_path = Path(file_path)
    extension = file_path.suffix.lower()

    if extension == ".txt":
        return load_txt(file_path)

    if extension == ".pdf":
        return load_pdf(file_path)

    if extension == ".docx":
        return load_docx(file_path)

    if extension in [".png", ".jpg", ".jpeg"]:
        return load_image(file_path)

    if extension in [".csv", ".xlsx", ".xls"]:
        return load_csv_or_excel(file_path)

    raise ValueError(f"Unsupported file type: {extension}")
