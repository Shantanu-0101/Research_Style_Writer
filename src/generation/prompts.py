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
    web_results: list[dict],
    style_notes: str,
    extra_instructions: str = "",
    previous_sections_summary: str = "",
) -> str:
    
    if web_results:
        content_block = "\n\n".join(
            f"[{r['title']}]({r['url']})\n{r['content'][:500]}" for r in web_results
        )
        content_instruction = (
            "Use the following real, current sources for factual content. "
            "You may reference them by publication/title if natural, but do NOT invent "
            "author names, years, or citations that are not explicitly given below."
        )
    else:
        content_block = "(no external sources retrieved)"
        content_instruction = (
            "No external sources were retrieved. Write using general domain knowledge, "
            "but do NOT fabricate specific citations, author names, or years. "
            "State claims generally rather than attributing them to invented sources."
        )

    guidance = SECTION_GUIDANCE.get(section.lower(), "Write in a clear academic style.")
    previous_context_block = (
        f"\nContext from sections already written earlier in this paper "
        f"(stay consistent, don't repeat content or phrasing):\n{previous_sections_summary}\n"
        if previous_sections_summary else ""
    )

    return f"""You are an academic writing assistant. Write the "{section}" section
for a research paper titled: "{topic}".

Section guidance: {guidance}
{previous_context_block}
Match this writing style (learned from reference papers, NOT the topic source):
{style_notes}

{content_instruction}

{content_block}

{f"Additional instructions: {extra_instructions}" if extra_instructions else ""}

Write only the section content, in formal academic prose, without a heading.
Vary sentence length and structure naturally — avoid stacking multiple transition
words (however, furthermore, moreover, therefore) in the same paragraph.
"""
    return prompt


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

Rewrite the section to fix ONLY these issues. Preserve everything else that
already works well — don't rewrite from scratch, don't change claims that
weren't flagged, and don't introduce new citations or facts not already
present in the original draft.

Match this writing style:
{style_notes}

Additional guidance for natural phrasing: avoid stacking multiple transition
words in one paragraph, vary sentence length and structure rather than
applying a uniform rhythm, and never invent specific details, anecdotes, or
citations to fill gaps — if something is genuinely missing, state the claim
more generally instead of fabricating a source for it.

Write only the revised section content, without a heading.
"""