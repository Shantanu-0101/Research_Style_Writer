from src.generation.prompts import build_rewrite_prompt
from src.llm.llm_factory import get_llm


def rewrite_section(topic: str, section: str, original_text: str, issues: list[str], style_profile) -> str:
    """
    Takes a section's text plus the Evaluator's specific complaints and
    asks the LLM to fix exactly those issues — not regenerate from scratch.
    Targeted revision keeps whatever was already working, instead of
    risking new problems by starting over each time.
    """
    prompt = build_rewrite_prompt(
        topic=topic,
        section=section,
        original_text=original_text,
        issues=issues,
        style_notes=style_profile.to_prompt_text(),
    )
    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content