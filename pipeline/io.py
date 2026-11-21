"""Read CV (PDF/TXT) and job description (TXT path or literal text)."""
from pathlib import Path


class InputError(ValueError):
    pass


def _pdf_text(path: Path) -> str:
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception:  # noqa: BLE001 - fall back to pypdf on any pdfplumber failure
        text = ""
    if not text.strip():
        try:
            from pypdf import PdfReader
            text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        except Exception as exc:  # noqa: BLE001
            raise InputError(f"Cannot read PDF {path}: {exc}") from exc
    if not text.strip():
        raise InputError(f"No extractable text in {path} (scanned/image PDF? OCR is out of scope)")
    return text


def read_cv(path) -> str:
    path = Path(path)
    if not path.is_file():
        raise InputError(f"CV file not found: {path}")
    if path.suffix.lower() == ".pdf":
        return _pdf_text(path)
    if path.suffix.lower() == ".txt":
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise InputError(f"CV file is empty: {path}")
        return text
    raise InputError(f"Unsupported CV format {path.suffix!r}; use .pdf or .txt")


def read_job(value: str) -> str:
    path = Path(value)
    try:
        is_file = path.suffix.lower() == ".txt" and path.is_file()
    except OSError:  # very long literal text is not a valid path
        is_file = False
    text = path.read_text(encoding="utf-8") if is_file else value
    if not text.strip():
        raise InputError("Job description is empty")
    return text


def list_cvs(directory) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise InputError(f"CV directory not found: {directory}")
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in (".pdf", ".txt"))
    if not files:
        raise InputError(f"No .pdf/.txt CVs in {directory}")
    return files
