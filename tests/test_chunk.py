from src.core.ingestion.chunker import chunk_pages

def test_chunk_pages_splits_long_text():
    pages = [{"text": "word " * 500, "source": "test.pdf", "page": 1}]
    chunks = chunk_pages(pages)
    assert len(chunks) > 1
    assert all(c["source"] == "test.pdf" for c in chunks)