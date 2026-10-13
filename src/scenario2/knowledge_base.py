"""
Knowledge Base builder and RAG retrieval for Scenario 2.

Wraps Embedder + FaissStore to:
  1. Build a vector index from all ingested materials (multi-format)
  2. Retrieve relevant context per section via semantic search
  3. Identify knowledge gaps where web search is needed
"""

from src.core.ingestion.chunker import chunk_pages
from src.core.embeddings.embedder import Embedder
from src.core.vectorstore.faiss_store import FaissStore
from src.core.config import settings


class KnowledgeBase:

    def __init__(self):
        self.embedder = Embedder()
        self.store = FaissStore()
        self.chunks: list[dict] = []
        self._built = False

    def build(self, pages: list[dict]) -> dict:
        """
        Chunks all ingested pages, embeds them, and builds the FAISS index.
        Returns stats about what was indexed.
        """
        self.chunks = chunk_pages(pages)
        if not self.chunks:
            return {"total_chunks": 0, "sources": []}

        texts = [c["text"] for c in self.chunks]
        vectors = self.embedder.embed_documents(texts)
        self.store.build(self.chunks, vectors)
        self._built = True

        sources = list({c["source"] for c in self.chunks})
        return {
            "total_chunks": len(self.chunks),
            "sources": sources,
        }

    def retrieve(self, query: str, k: int = None) -> list[dict]:
        """Semantic search over the knowledge base. Returns top-k relevant chunks."""
        if not self._built:
            return []
        k = k or settings.top_k
        query_vector = self.embedder.embed_query(query)
        return self.store.search(query_vector, k=k)

    def retrieve_for_section(self, topic: str, section: str, keywords: str = "") -> list[dict]:
        """
        Build a section-aware query and retrieve relevant KB chunks.
        Returns chunks with text, source, and page metadata.
        """
        query_parts = [topic]
        if keywords:
            kw_list = [k.strip() for k in keywords.split(",") if k.strip()][:4]
            query_parts.extend(kw_list)

        section_terms = {
            "abstract": "summary overview contributions",
            "introduction": "background motivation problem statement",
            "literature review": "related work prior research survey",
            "methodology": "approach method framework technique",
            "results": "findings evaluation performance metrics",
            "conclusion": "summary implications future work",
        }
        query_parts.append(section_terms.get(section.lower(), section))

        query = " ".join(query_parts)
        return self.retrieve(query, k=settings.top_k)

    def assess_coverage(self, topic: str, sections: list[str], keywords: str = "") -> dict:
        """
        Checks how well the KB covers each section.
        Returns per-section coverage scores and gap indicators.
        """
        if not self._built:
            return {s: {"chunks_found": 0, "has_gap": True} for s in sections}

        coverage = {}
        for section in sections:
            if section.lower() == "references":
                coverage[section] = {"chunks_found": 0, "has_gap": False}
                continue

            chunks = self.retrieve_for_section(topic, section, keywords)
            total_words = sum(len(c["text"].split()) for c in chunks)

            # A section with fewer than 200 words of KB material needs web supplementation
            has_gap = total_words < 200
            coverage[section] = {
                "chunks_found": len(chunks),
                "total_words": total_words,
                "has_gap": has_gap,
                "sources": list({c["source"] for c in chunks}),
            }

        return coverage

    def save(self):
        if self._built:
            self.store.save()

    def load(self):
        self.store.load()
        self.chunks = self.store.chunks
        self._built = True
