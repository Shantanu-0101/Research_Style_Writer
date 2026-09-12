SECTION_GUIDANCE = {
    "introduction": "Set the context, motivate the problem, and state the paper's contribution.",
    "literature review": "Summarize and synthesize related work, grouping similar approaches and noting gaps.",
    "methodology": "Describe the approach/method in a structured, step-by-step academic manner.",
    "results": "Present findings objectively, referencing data/metrics where relevant.",
    "conclusion": "Summarize contributions, implications, and possible future work.",
}


def build_generation_prompt(
    topic: str,
    section: str,
    retrieved_chunks: list[dict],
    style_notes: str,
    extra_instructions: str = "",
) -> str:
    
    context_text = "\n\n".join(
        f"[Source: {c['source']}, page {c['page']}]\n{c['text']}"
        for c in retrieved_chunks
    )

    guidance = SECTION_GUIDANCE.get(section.lower(), "Write in a clear academic style.")

    prompt = f"""You are an academic writing assistant. Write the "{section}" section
for a research paper titled: "{topic}".

Section guidance: {guidance}

Match this writing style, based on analysis of reference papers:
{style_notes}

Use the following reference material for factual grounding. Do not copy
sentences verbatim — synthesize and paraphrase in your own academic voice:

{context_text}

{f"Additional instructions: {extra_instructions}" if extra_instructions else ""}

Write only the section content, in formal academic prose, without a heading.
"""
    return prompt