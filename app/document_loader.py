"""
Document loading and text extraction module.
Supports TXT, PDF, and DOCX file formats.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def extract_text_from_txt(file_path: str) -> str:
    """Extract text content from a plain text file.

    Args:
        file_path: Path to the text file.

    Returns:
        The raw text content of the file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text content from a PDF file.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Concatenated text from all pages.
    """
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    return "\n".join(pages)


def extract_text_from_docx(file_path: str) -> str:
    """Extract text content from a DOCX file.

    Args:
        file_path: Path to the DOCX file.

    Returns:
        Concatenated text from all paragraphs.
    """
    from docx import Document

    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def load_document(file_path: str) -> Optional[str]:
    """Load and extract text from a document file.

    Args:
        file_path: Path to the document file.

    Returns:
        Extracted text content, or None if the format is unsupported.
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        logger.warning(
            "[document_loader] unsupported file format - path=%s, extension=%s",
            file_path,
            extension,
        )
        return None

    logger.info(
        "[document_loader] loading document - path=%s, extension=%s",
        file_path,
        extension,
    )

    extractors = {
        ".txt": extract_text_from_txt,
        ".pdf": extract_text_from_pdf,
        ".docx": extract_text_from_docx,
    }

    try:
        text = extractors[extension](file_path)
        logger.info(
            "[document_loader] document loaded - path=%s, characters=%d",
            file_path,
            len(text),
        )
        return text
    except Exception as e:
        logger.error(
            "[document_loader] failed to load document - path=%s, error=%s",
            file_path,
            str(e),
        )
        return None
