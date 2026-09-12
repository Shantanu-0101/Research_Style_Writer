from src.generation.prompts import build_generation_prompt
from src.style.style_profiler import profile_style
from src.llm.llm_factory import get_llm


def generate_section(
    topic: str,
    section: str,
    embedder,
    faiss_store,
    extra_instructions: str = "",
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
    )

    llm = get_llm()
    response = llm.invoke(prompt)

    sources_used = list({(c["source"], c["page"]) for c in retrieved_chunks})

    return {
        "text": response.content,
        "sources": sources_used,
    }