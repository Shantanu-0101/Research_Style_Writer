from src.generation.prompts import build_generation_prompt
from src.style.style_analyzer import StyleProfile
from src.llm.llm_factory import get_llm
from src.generation.rewriter import rewrite_section
from src.evaluation.evaluator import evaluate_section
from src.retrieval.web_search import search_topic_content

DEFAULT_SECTIONS = ["Introduction", "Literature Review", "Methodology", "Results", "Conclusion"]


def generate_section(
    topic: str,
    section: str,
    style_profile: StyleProfile,
    web_results: list[dict] = None,
    extra_instructions: str = "",
    previous_sections_summary: str = "",
) -> dict:
    if web_results is None:
        web_results = search_topic_content(topic, section)

    prompt = build_generation_prompt(
        topic=topic,
        section=section,
        web_results=web_results,
        style_notes=style_profile.to_prompt_text(),
        extra_instructions=extra_instructions,
        previous_sections_summary=previous_sections_summary,
    )

    llm = get_llm()
    response = llm.invoke(prompt)
    sources_used = [{"title": r["title"], "url": r["url"]} for r in web_results]
    return {"text": response.content, "sources": sources_used}


def generate_reviewed_section(
    topic: str,
    section: str,
    style_profile: StyleProfile,
    web_results: list[dict] = None,
    extra_instructions: str = "",
    previous_sections_summary: str = "",
    max_attempts: int = 3,
) -> dict:
    """
    Runs generate -> evaluate -> rewrite, looping until the Evaluator
    passes the section or max_attempts is reached.
    """
    result = generate_section(
        topic=topic,
        section=section,
        style_profile=style_profile,
        web_results=web_results,               # now threaded through, not dropped
        extra_instructions=extra_instructions,
        previous_sections_summary=previous_sections_summary,
    )

    attempt = 1
    evaluation = evaluate_section(result["text"], result["sources"], style_profile, previous_sections_summary)

    while not evaluation.passed and attempt < max_attempts:
        result["text"] = rewrite_section(
            topic=topic,
            section=section,
            original_text=result["text"],
            issues=evaluation.issues,
            style_profile=style_profile,
        )
        attempt += 1
        evaluation = evaluate_section(result["text"], result["sources"], style_profile, previous_sections_summary)

    return {
        "text": result["text"],
        "sources": result["sources"],
        "attempts": attempt,
        "passed": evaluation.passed,
        "remaining_issues": evaluation.issues,
    }


def _summarize_for_context(section_name: str, text: str, max_words: int = 60) -> str:
    words = text.split()
    snippet = " ".join(words[:max_words])
    return f"- {section_name}: {snippet}{'...' if len(words) > max_words else ''}"


def generate_full_paper(topic, style_profile, sections=None, extra_instructions="", max_attempts: int = 3):
    sections = sections or DEFAULT_SECTIONS
    running_summary_parts = []
    used_urls = set()

    for section in sections:
        web_results = search_topic_content(topic, section)
        fresh_results = [r for r in web_results if r["url"] not in used_urls]
        for r in fresh_results:
            used_urls.add(r["url"])

        result = generate_reviewed_section(
            topic=topic,
            section=section,
            style_profile=style_profile,
            web_results=fresh_results,                                    # fixed: was dropped
            extra_instructions=extra_instructions,
            previous_sections_summary="\n".join(running_summary_parts),   # fixed: was dropped
            max_attempts=max_attempts,
        )
        running_summary_parts.append(_summarize_for_context(section, result["text"]))
        yield section, result