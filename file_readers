"""Text extraction from uploaded files (PDF, PPTX, XLSX, DOCX).

This module turns a file uploaded through Streamlit into plain text that
can be translated. It supports four formats and enforces a maximum file
size. Only text is extracted: images, charts and embedded media are
ignored, so an image-heavy PDF may yield little or no text.

DESIGN NOTES
============
- Extraction functions take a file-like object (what Streamlit's
  st.file_uploader returns) and return a string. They contain no
  Streamlit UI calls, so they stay testable and reusable.
- The single public entry point, extract_text(), dispatches on the
  file's MIME type and applies the size limit. It raises
  FileTooLargeError or UnsupportedFileError for the two expected error
  conditions, which the UI layer is responsible for catching and
  presenting to the user.

ABOUT EXTRACTED TEXT QUALITY
============================
Text pulled from PDFs and slide decks is often dense or out of order:
lines can run together, reading order may be wrong, headers and footers
may be interleaved with body text. This is inherent to those formats.
The translation pipeline's refinement/upgrade step is where layout is
cleaned up; extraction only aims to recover the words.
"""

from io import BytesIO

import fitz  # PyMuPDF
from docx import Document
from openpyxl import load_workbook
from pptx import Presentation


# Maximum accepted upload size. Kept deliberately modest: the extracted
# text is sent to a language model, so very large files mean high cost
# and long latency. Adjust here if the policy changes.
MAX_FILE_SIZE_MB = 20

# MIME types produced by st.file_uploader for each supported extension.
_MIME_PDF = "application/pdf"
_MIME_PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
_MIME_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_MIME_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class FileTooLargeError(Exception):
    """Raised when an uploaded file exceeds MAX_FILE_SIZE_MB."""


class UnsupportedFileError(Exception):
    """Raised when an uploaded file's type is not one we can read."""


# ---------------------------------------------------------------------------
# Per-format extractors
# ---------------------------------------------------------------------------

def read_pdf(file) -> str:
    """Extract text from a PDF file.

    Args:
        file: A file-like object (e.g. the object returned by
            st.file_uploader). Its bytes are read into memory.

    Returns:
        The concatenated text of every page. Pages that contain only
        images contribute nothing.
    """
    text_parts = []
    bytes_stream = BytesIO(file.read())
    with fitz.open(stream=bytes_stream, filetype="pdf") as document:
        for page in document:
            text_parts.append(page.get_text())
    return "".join(text_parts)


def read_pptx(file) -> str:
    """Extract text from a PowerPoint presentation.

    Walks every slide and collects the text of every shape that has a
    text frame (titles, bullet points, text boxes).

    Args:
        file: A file-like object for the .pptx file.

    Returns:
        The collected text, with a space between successive shapes.
    """
    text_parts = []
    presentation = Presentation(file)
    for slide in presentation.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text_parts.append(shape.text)
    return " ".join(text_parts)


def read_excel(file) -> str:
    """Extract cell values from an Excel workbook.

    Reads every sheet, every row, every cell, converting each value to a
    string. Empty cells become the string "None"; this is acceptable for
    translation input but worth knowing when inspecting output.

    Args:
        file: A file-like object for the .xlsx file.

    Returns:
        All cell values joined by spaces, in row-major order across all
        sheets.
    """
    text_parts = []
    workbook = load_workbook(filename=file)
    for sheet in workbook:
        for row in sheet.iter_rows(values_only=True):
            for cell in row:
                text_parts.append(str(cell))
    return " ".join(text_parts)


def read_docx(file) -> str:
    """Extract text from a Word document.

    Collects the text of every paragraph. Tables and text boxes are not
    extracted by this simple reader.

    Args:
        file: A file-like object for the .docx file.

    Returns:
        The document's paragraph text, joined by spaces.
    """
    document = Document(file)
    return " ".join(paragraph.text for paragraph in document.paragraphs)


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------

# Maps each supported MIME type to its extractor function.
_EXTRACTORS = {
    _MIME_PDF: read_pdf,
    _MIME_PPTX: read_pptx,
    _MIME_XLSX: read_excel,
    _MIME_DOCX: read_docx,
}


def extract_text(uploaded_file) -> str:
    """Extract plain text from a Streamlit-uploaded file.

    This is the single entry point the UI should call. It checks the
    file size, selects the right extractor based on the file's MIME
    type, and returns the extracted text.

    Args:
        uploaded_file: The object returned by st.file_uploader. Must
            expose `.size` (bytes), `.type` (MIME string) and the usual
            file-like read interface.

    Returns:
        The extracted text. May be empty if the file contained no
        extractable text (e.g. a scanned, image-only PDF).

    Raises:
        FileTooLargeError: If the file exceeds MAX_FILE_SIZE_MB. The
            message is suitable for display to the user.
        UnsupportedFileError: If the file's MIME type is not one of the
            four supported formats. The message is suitable for display.
    """
    size_limit_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if uploaded_file.size > size_limit_bytes:
        raise FileTooLargeError(
            f"File too large. The maximum size is {MAX_FILE_SIZE_MB} MB."
        )

    extractor = _EXTRACTORS.get(uploaded_file.type)
    if extractor is None:
        raise UnsupportedFileError(
            "Unsupported file type. Please upload a PDF, PPTX, XLSX or DOCX file."
        )

    return extractor(uploaded_file)
