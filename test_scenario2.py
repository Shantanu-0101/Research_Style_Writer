"""Scenario 2 import and sanity test."""
import io

# 1. Ingestion module
from src.scenario2.ingestion import (
    extract_from_upload, extract_from_uploads, fetch_url_content,
    _extract_docx, _extract_pptx, _extract_txt, _extract_tex,
    SUPPORTED_EXTENSIONS,
)
print("Supported extensions:", SUPPORTED_EXTENSIONS)

# Test TXT extraction
txt_bytes = b"This is a test document with enough words to be meaningful for the knowledge base builder."
pages = _extract_txt(txt_bytes, "test.txt")
print(f"TXT extraction: {len(pages)} page(s), {len(pages[0]['text'].split())} words")
assert pages[0]["source"] == "test.txt"

# Test TEX extraction
tex_bytes = br"""
\documentclass{article}
\begin{document}
\section{Introduction}
This is a LaTeX document about machine learning approaches to natural language processing.
We present a novel framework for text classification.

\section{Methodology}
Our approach uses transformer-based architectures with attention mechanisms
to capture long-range dependencies in text sequences.
\end{document}
""".strip()
tex_pages = _extract_tex(tex_bytes, "paper.tex")
print(f"TEX extraction: {len(tex_pages)} page(s)")
# Check that LaTeX commands were stripped
for p in tex_pages:
    assert "\\section" not in p["text"]
    assert "\\documentclass" not in p["text"]
print("TEX command stripping: OK")

# 2. Knowledge Base
from src.scenario2.knowledge_base import KnowledgeBase
kb = KnowledgeBase()
# Build with simple test data
test_pages = [
    {"text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data. Deep learning uses neural networks with multiple layers.", "source": "ml_notes.txt", "page": 1},
    {"text": "Natural language processing deals with the interaction between computers and human language. Transformers have revolutionized NLP tasks.", "source": "nlp_paper.pdf", "page": 1},
    {"text": "Knowledge graphs represent structured information as entities and relationships. They enable reasoning over connected data.", "source": "kg_slides.pptx", "page": 1},
]
stats = kb.build(test_pages)
print(f"KB built: {stats['total_chunks']} chunks from {len(stats['sources'])} sources")
assert stats["total_chunks"] > 0

# Test retrieval
results = kb.retrieve("deep learning neural networks")
print(f"KB retrieval: {len(results)} result(s)")
assert len(results) > 0

# Test section retrieval
section_results = kb.retrieve_for_section("Machine Learning", "Introduction", "deep learning, NLP")
print(f"KB section retrieval: {len(section_results)} result(s)")

# Test coverage assessment
coverage = kb.assess_coverage("Machine Learning", ["Introduction", "Methodology", "References"], "deep learning")
print(f"Coverage assessment: {len(coverage)} sections checked")
assert "References" in coverage
assert coverage["References"]["has_gap"] is False
print("Coverage analysis: OK")

# 3. Prompts
from src.scenario2.prompts import build_generation_prompt, build_rewrite_prompt
prompt = build_generation_prompt(
    topic="Machine Learning in Healthcare",
    section="Introduction",
    kb_context=[{"text": "ML has been applied...", "source": "notes.txt", "page": 1}],
    web_results=[{"title": "ML Health", "url": "https://example.com", "content": "Recent advances..."}],
    style_notes="Formal academic tone",
    keywords="machine learning, healthcare",
    has_gap=True,
)
assert "PRIMARY SOURCE" in prompt
assert "SUPPLEMENTARY SOURCE" in prompt
assert "knowledge gaps" in prompt.lower() or "gap" in prompt.lower()
print(f"Prompt generation: OK ({len(prompt)} chars)")

# 4. Generator imports
from src.scenario2.generator import (
    generate_section, generate_reviewed_section, generate_full_paper,
    check_post_humanization_quality, _build_image_query,
)
query = _build_image_query("AI in Healthcare", "Methodology", "deep learning, diagnosis")
assert "deep learning" in query
assert "framework" in query or "architecture" in query
print(f"Image query builder: '{query}'")

# 5. Quality check
quality = check_post_humanization_quality("This is a test. But scale matters. The results show improvement.")
print(f"Quality check: {len(quality['metrics'])} metrics")

print("\nAll Scenario 2 imports and tests PASSED")
