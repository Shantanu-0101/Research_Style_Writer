from src.generation.prompts import build_generation_prompt
from src.style.style_profiler import profile_style
from src.llm.llm_factory import get_llm

DEFAULT_SECTIONS = ["Introduction", "Literature Review", "Methodology", "Results", "Conclusion"]

def generate_section(
    topic: str,
    section: str,
    embedder,
    faiss_store,
    extra_instructions: str = "",
    previous_sections_summary="",
) -> dict:
    
    query = f"{topic} {section}"
    query_vector = embedder.embed_query(query)
    retrieved_chunks = faiss_store.search(query_vector)

    style_notes = profile_style(retrieved_chunks)

    prompt = build_generation_prompt(
        topic=topic,
        section=section,
        retrieved_chunks=retrieved_chunks,
        style_notes=style_notes,
        extra_instructions=extra_instructions,
        previous_sections_summary=previous_sections_summary,
    )

    llm = get_llm()
    response = llm.invoke(prompt)

    sources_used = list({(c["source"], c["page"]) for c in retrieved_chunks})

    return {
        "text": response.content,
        "sources": sources_used,
    }

def _summarize_for_context(section_name: str, text: str, max_words: int = 60) -> str:
    """
    Compresses a completed section down to a short pointer for later
    sections to reference — not the full text, just enough to keep
    the paper internally consistent without bloating every prompt.
    """
    words = text.split()
    snippet = " ".join(words[:max_words])
    return f"- {section_name}: {snippet}{'...' if len(words) > max_words else ''}"


def generate_full_paper(topic, embedder, faiss_store, sections=None, extra_instructions=""):
    """
    Generates each section in order, yielding results one at a time
    so the caller (Streamlit UI) can render them as they complete,
    instead of waiting for the whole paper before showing anything.
    """
    sections = sections or DEFAULT_SECTIONS
    running_summary_parts = []

    for section in sections:
        result = generate_section(
            topic=topic,
            section=section,
            embedder=embedder,
            faiss_store=faiss_store,
            extra_instructions=extra_instructions,
            previous_sections_summary="\n".join(running_summary_parts),
        )
        running_summary_parts.append(_summarize_for_context(section, result["text"]))
        yield section, result