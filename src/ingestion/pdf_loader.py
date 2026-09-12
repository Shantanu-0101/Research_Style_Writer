import pymupdf as fitz

def extract_text_from_pdf(file_bytes: bytes, filename: str) -> list[dict]:
    
    pages = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception:
        return pages

    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text").strip()
        if text:  # skip blank/scanned pages with no extractable text
            pages.append({
                "text": text,
                "source": filename,
                "page": page_num + 1,
            })

    doc.close()
    return pages


def extract_text_from_uploads(uploaded_files) -> tuple[list[dict], list[str]]:
    
    all_pages = []
    skipped = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        pages = extract_text_from_pdf(file_bytes, uploaded_file.name)

        if pages:
            all_pages.extend(pages)
        else:
            skipped.append(uploaded_file.name)

    return all_pages, skipped