
from src.ingestion.pdf_loader import extract_text_from_pdf
from src.ingestion.chunker import chunk_pages
from src.embeddings.embedder import Embedder
from src.vectorstore.faiss_store import FaissStore
from src.llm.llm_factory import get_llm

# Grab one real PDF from disk to test with (no Streamlit needed yet)
# with open("data/uploads/Attention is all you need.pdf", "rb") as f:
#     file_bytes = f.read()

# pages = extract_text_from_pdf(file_bytes, "Attention is all you need.pdf")
# print(f"Extracted {len(pages)} pages")
# print(pages[0]["text"][:200])  # peek at first 200 chars of page 1

# chunks = chunk_pages(pages)
# print(f"Split into {len(chunks)} chunks")
# print(chunks[0])

# embedder = Embedder()
# vectors = embedder.embed_documents([c["text"] for c in chunks[:3]])
# print(f"Got {len(vectors)} vectors, each of length {len(vectors[0])}")

# store = FaissStore()
# all_vectors = embedder.embed_documents([c["text"] for c in chunks])
# store.build(chunks, all_vectors)

# query_vec = embedder.embed_query("what methodology did they use")
# results = store.search(query_vec, k=3)
# for r in results:
#     print(r["source"], r["page"], r["text"][:100])

llm = get_llm()
response = llm.invoke("Say hello in one sentence.")
print(response.content)