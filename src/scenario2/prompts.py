"""
Scenario 2 prompts — hybrid KB + web context generation.

Key difference from Scenario 1:
  - User's own materials (KB context) are the PRIMARY source
  - Web results fill gaps where user materials are insufficient
  - Prompt explicitly distinguishes between the two sources
"""

SECTION_GUIDANCE = {
    "abstract": "Provide a concise summary of the research problem, methodology, key findings, and contributions in 150-250 words.",
    "introduction": "Set the context, motivate the problem, and state the paper's contribution.",
    "literature review": "Summarize and synthesize related work, grouping similar approaches and noting gaps.",
    "methodology": "Describe the approach/method in a structured, step-by-step academic manner.",
    "results": "Present findings objectively, referencing data/metrics where relevant.",
    "conclusion": "Summarize contributions, implications, and possible future work.",
    "references": "List all cited works in a consistent academic citation format (APA style).",
}


def build_generation_prompt(
    topic: str,
    section: str,
    kb_context: list[dict],
    web_results: list[dict],
    style_notes: str = "",
    style_profile=None,
    keywords: str = "",
    extra_instructions: str = "",
    previous_sections_summary: str = "",
    include_visuals: bool = True,
    has_gap: bool = False,
) -> str:

    # Primary source: user's research materials from knowledge base
    if kb_context:
        kb_block = "\n\n".join(
            f"[From: {c['source']}, p.{c['page']}]\n{c['text'][:600]}" for c in kb_context
        )
        kb_instruction = (
            "PRIMARY SOURCE — The following excerpts are from the user's own research materials "
            "(uploaded papers, notes, slides, LaTeX sources). Synthesize and build upon these as "
            "the foundation of this section. Cite the source document when referencing specific claims."
        )
    else:
        kb_block = "(No user-provided materials found for this section)"
        kb_instruction = (
            "No user materials were found for this section. Rely on the web sources below."
        )

    # Secondary source: web search results (gap filling)
    if web_results:
        web_block = "\n\n".join(
            f"[{r['title']}]({r['url']})\n{r['content'][:500]}" for r in web_results
        )
        if has_gap:
            web_instruction = (
                "SUPPLEMENTARY SOURCE — The user's materials did not fully cover this section. "
                "Use these web sources to fill knowledge gaps. Reference them by publication/title "
                "if natural, but do NOT invent author names, years, or citations."
            )
        else:
            web_instruction = (
                "ADDITIONAL CONTEXT — The user's materials cover this section well. These web sources "
                "provide supplementary context. Use them sparingly to enrich the section, but prioritize "
                "the user's own materials above."
            )
    else:
        web_block = "(No web sources retrieved)"
        web_instruction = "No supplementary web sources were retrieved."

    if style_profile is not None:
        style_notes = style_profile.to_prompt_text()

    guidance = SECTION_GUIDANCE.get(section.lower(), "Write in a clear academic style.")
    previous_context_block = (
        f"\nContext from sections already written earlier in this paper "
        f"(stay consistent, don't repeat content or phrasing):\n{previous_sections_summary}\n"
        if previous_sections_summary else ""
    )

    # Dynamic word count from reference papers
    ratios = {
        "abstract": 0.05,
        "introduction": 0.15,
        "literature review": 0.25,
        "methodology": 0.25,
        "results": 0.20,
        "conclusion": 0.10,
        "references": 0.0,
    }
    ratio = ratios.get(section.lower(), 0.15)
    total_target = getattr(style_profile, "avg_paper_word_count", 2500) if style_profile else 2500
    target_words = int(total_target * ratio)

    length_instruction = ""
    if section.lower() == "abstract":
        length_instruction = "\n## Length Requirement\nWrite a concise abstract of 150-250 words.\n"
    elif section.lower() != "references":
        length_instruction = f"""
## Length & Depth Requirement
This section MUST be comprehensive and academically deep.
Target: at least {target_words} words for this section.
Break into 2-4 logical subsections using markdown subheadings.
For each subsection: write multiple detailed paragraphs with technical depth.
"""

    structural_instructions = ""
    if section.lower() not in ("abstract", "references"):
        structural_instructions = """
## Structural Variation (CRITICAL for authentic prose)
1. SENTENCE-INITIAL DIVERSITY: At least 40% of sentences should open with subordinate clauses, prepositional phrases, adverbs, or gerund phrases — not just the subject.
2. PARAGRAPH STRUCTURE VARIETY: Not every paragraph should start with a topic sentence. Occasionally open with a specific observation, a question, or a concession.
3. SENTENCE LENGTH VARIATION: Mix very short sentences (4-8 words) with longer compound-complex sentences (25-35 words).
4. AVOID AI PARAGRAPH TEMPLATE: Do NOT follow "general claim → 3 evidence pieces → concluding generalization" in every paragraph.
"""

    visual_instructions = ""
    if include_visuals and section.lower() not in ("references", "abstract"):
        visual_instructions = f"""
Include exactly ONE relevant academic figure or diagram.
Use this syntax: ![Clear descriptive caption](image_query: "3-5 specific keywords")
Place it at the most logically relevant point. Use specific search terms matching this section's content.
"""

    keywords_block = ""
    if keywords:
        keywords_block = f"""
## Keywords (weave these naturally into the text)
{keywords}
"""

    statistics_block = ""
    if section.lower() in ("results", "methodology"):
        statistics_block = """
## Statistical Analysis Requirement
If this topic involves quantitative data, present appropriate statistical analysis,
confidence intervals, and significance measures where relevant.
"""

    return f"""You are an academic writing assistant. Write the "{section}" section
for a research paper titled: "{topic}".

Section guidance: {guidance}
{previous_context_block}

{length_instruction}

{structural_instructions}

Match this writing style (learned from reference papers):
{style_notes}

{keywords_block}

## User's Research Materials (PRIMARY)
{kb_instruction}

{kb_block}

## Web Sources (SUPPLEMENTARY)
{web_instruction}

{web_block}

{visual_instructions}

{statistics_block}

{f"Additional instructions: {extra_instructions}" if extra_instructions else ""}

Write only the section content in formal academic prose. Do not include a top-level heading for the section name.
Use subheadings as instructed.
CRITICAL: Do NOT use: delves, tapestry, beacon, landscape (figurative), realm, paradigm shift, groundbreaking, holistic, cornerstone, interplay, vibrant, fosters, underscores, pivotal, crucial, testament, unprecedented, transformative, seamless integration.
"""


def build_rewrite_prompt(
    topic: str,
    section: str,
    original_text: str,
    issues: list[str],
    style_notes: str,
) -> str:
    issues_block = "\n".join(f"- {issue}" for issue in issues)

    return f"""You are revising the "{section}" section of a research paper titled "{topic}".

Here is the current draft:
---
{original_text}
---

An automated review flagged these specific issues:
{issues_block}

Rewrite the section to fix ONLY these issues. Preserve everything else.

Match this writing style:
{style_notes}

CRITICAL:
- Preserve sentence-initial variety, sentence length variation, and paragraph structure variety
- Do NOT use: delves, tapestry, beacon, landscape (figurative), realm, paradigm shift, groundbreaking, holistic, cornerstone, interplay, vibrant, fosters, underscores, pivotal, crucial, testament, unprecedented, transformative, seamless integration
- Never invent specific details, anecdotes, or citations

Write only the revised section content, without a top-level heading.
"""
