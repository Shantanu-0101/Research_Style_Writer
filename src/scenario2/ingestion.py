"""
Multi-format ingestion for Scenario 2.

Supported formats:
  - PDF  (.pdf)   — reuses core pdf_loader + VLM fallback
  - DOCX (.docx)  — python-docx paragraph extraction
  - PPTX (.pptx)  — python-pptx slide text extraction
  - TXT  (.txt)   — direct read
  - TEX  (.tex)   — LaTeX command stripping → plain text
  - URLs          — Tavily content fetch or direct requests fallback
"""

import re
import io
import requests
from pathlib import Path

from src.core.ingestion.pdf_loader import extract_text_from_pdf
from src.core.config import settings


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".tex"}


def _extract_docx(file_bytes: bytes, filename: str) -> list[dict]:
    from docx import Document
    doc = Document(io.BytesIO(file_bytes))
    pages = []
    current_page_text = []
    para_count = 0

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        current_page_text.append(text)
        para_count += 1
        # Group roughly 15 paragraphs per "page" for chunking consistency
        if para_count >= 15:
            pages.append({
                "text": "\n\n".join(current_page_text),
                "source": filename,
                "page": len(pages) + 1,
            })
            current_page_text = []
            para_count = 0

    if current_page_text:
        pages.append({
            "text": "\n\n".join(current_page_text),
            "source": filename,
            "page": len(pages) + 1,
        })
    return pages


def _extract_pptx(file_bytes: bytes, filename: str) -> list[dict]:
    from pptx import Presentation
    prs = Presentation(io.BytesIO(file_bytes))
    pages = []

    for slide_num, slide in enumerate(prs.slides, 1):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    line = para.text.strip()
                    if line:
                        texts.append(line)
        if texts:
            pages.append({
                "text": "\n".join(texts),
                "source": filename,
                "page": slide_num,
            })
    return pages


def _extract_txt(file_bytes: bytes, filename: str) -> list[dict]:
    text = file_bytes.decode("utf-8", errors="replace").strip()
    if not text:
        return []

    # Split into ~1000 word blocks to match chunking expectations
    words = text.split()
    block_size = 800
    pages = []
    for i in range(0, len(words), block_size):
        block = " ".join(words[i : i + block_size])
        pages.append({
            "text": block,
            "source": filename,
            "page": (i // block_size) + 1,
        })
    return pages


# Patterns to strip from LaTeX
_TEX_STRIP = [
    (re.compile(r"\\begin\{(abstract|equation\*?|align\*?|figure|table|lstlisting|verbatim)\}.*?\\end\{\1\}", re.DOTALL), " "),
    (re.compile(r"\\(usepackage|documentclass|bibliographystyle|bibliography|label|ref|cite|citep|citet|eqref|url|href)\{[^}]*\}"), ""),
    (re.compile(r"\\(section|subsection|subsubsection|paragraph|chapter)\*?\{([^}]*)\}"), r"\n\n\2\n\n"),
    (re.compile(r"\\(textbf|textit|emph|underline|texttt)\{([^}]*)\}"), r"\2"),
    (re.compile(r"\\(title|author|date)\{([^}]*)\}"), r"\2"),
    (re.compile(r"\\[a-zA-Z]+\{[^}]*\}"), ""),
    (re.compile(r"\\[a-zA-Z]+"), ""),
    (re.compile(r"[{}]"), ""),
    (re.compile(r"%[^\n]*"), ""),
    (re.compile(r"\n{3,}"), "\n\n"),
]


def _extract_tex(file_bytes: bytes, filename: str) -> list[dict]:
    text = file_bytes.decode("utf-8", errors="replace")

    # Strip preamble (everything before \begin{document})
    doc_start = text.find(r"\begin{document}")
    if doc_start != -1:
        text = text[doc_start + len(r"\begin{document}"):]
    doc_end = text.find(r"\end{document}")
    if doc_end != -1:
        text = text[:doc_end]

    for pattern, replacement in _TEX_STRIP:
        text = pattern.sub(replacement, text)

    text = text.strip()
    if not text:
        return []

    # Split into page-like blocks by double newlines
    blocks = [b.strip() for b in text.split("\n\n") if b.strip() and len(b.strip()) > 30]
    pages = []
    current_block = []
    current_len = 0

    for block in blocks:
        current_block.append(block)
        current_len += len(block.split())
        if current_len >= 600:
            pages.append({
                "text": "\n\n".join(current_block),
                "source": filename,
                "page": len(pages) + 1,
            })
            current_block = []
            current_len = 0

    if current_block:
        pages.append({
            "text": "\n\n".join(current_block),
            "source": filename,
            "page": len(pages) + 1,
        })
    return pages


def extract_from_upload(file_bytes: bytes, filename: str) -> list[dict]:
    """Route a single uploaded file to the correct extractor based on extension."""
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes, filename)
    elif ext == ".docx":
        return _extract_docx(file_bytes, filename)
    elif ext == ".pptx":
        return _extract_pptx(file_bytes, filename)
    elif ext == ".txt":
        return _extract_txt(file_bytes, filename)
    elif ext == ".tex":
        return _extract_tex(file_bytes, filename)
    else:
        print(f"[ingestion] Unsupported format: {ext} ({filename})")
        return []


def extract_from_uploads(uploaded_files) -> tuple[list[dict], list[str]]:
    """Process multiple uploaded files of mixed formats. Returns (pages, skipped_filenames)."""
    all_pages = []
    skipped = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        pages = extract_from_upload(file_bytes, uploaded_file.name)
        if pages:
            all_pages.extend(pages)
        else:
            skipped.append(uploaded_file.name)

    return all_pages, skipped


def fetch_url_content(url: str) -> list[dict]:
    """Fetch content from a URL using Tavily (preferred) or direct requests fallback."""
    # Try Tavily first for clean content extraction
    if settings.tavily_api_key and settings.tavily_api_key.startswith("tvly-"):
        try:
            payload = {
                "api_key": settings.tavily_api_key,
                "query": url,
                "include_raw_content": True,
                "max_results": 1,
            }
            resp = requests.post("https://api.tavily.com/search", json=payload, timeout=15)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            for r in results:
                content = r.get("raw_content") or r.get("content", "")
                if content and len(content) > 100:
                    return [{
                        "text": content[:10000],
                        "source": url,
                        "page": 1,
                    }]
        except Exception as e:
            print(f"[ingestion] Tavily URL fetch failed: {e}")

    # Direct fetch fallback
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=12)
        resp.raise_for_status()
        from lxml import html as lxml_html
        tree = lxml_html.fromstring(resp.content)
        # Remove script and style tags
        for elem in tree.xpath("//script | //style | //nav | //footer | //header"):
            elem.getparent().remove(elem)
        text = tree.text_content()
        # Clean whitespace
        text = re.sub(r"\s+", " ", text).strip()
        if text and len(text) > 100:
            return [{
                "text": text[:10000],
                "source": url,
                "page": 1,
            }]
    except Exception as e:
        print(f"[ingestion] Direct URL fetch failed for {url}: {e}")

    return []
