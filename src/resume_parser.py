import io, re
import pdfplumber, docx

def extract_text_from_file(filename: str, file_bytes: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n".join(pages).strip()
    elif name.endswith(".docx"):
        d = docx.Document(io.BytesIO(file_bytes))
        return "\n".join([p.text for p in d.paragraphs]).strip()
    elif name.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")
    else:
        try:
            return file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            return ""

def guess_name_from_filename(filename: str) -> str:
    base = filename.rsplit("/", 1)[-1]
    base = base.rsplit(".", 1)[0]
    base = re.sub(r"[_\-]+", " ", base)
    base = re.sub(r"\s+", " ", base).strip()
    return base.title() or "Unknown"
