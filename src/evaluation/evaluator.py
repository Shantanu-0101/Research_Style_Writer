# src/evaluation/evaluator.py
import re
import statistics
from dataclasses import dataclass, field


@dataclass
class EvaluationResult:
    passed: bool
    issues: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.passed:
            return "No issues found."
        return "\n".join(f"- {issue}" for issue in self.issues)


CITATION_PATTERN = re.compile(r"\(([A-Z][a-zA-Z\-]+(?:\s(?:&|and)\s[A-Z][a-zA-Z\-]+)?,?\s*\d{4}[a-z]?)\)")


def check_fabricated_citations(text: str, sources_used: list[dict]) -> list[str]:
    """
    Flags (Author, Year)-style citations only when NO web sources were retrieved at all,
    meaning the LLM had nothing to ground facts in and may have hallucinated citations.
    When real sources exist, LLMs naturally adopt citation-style prose — that's acceptable.
    """
    if sources_used:
        # Sources were retrieved — citation-style phrasing is likely grounded, don't block
        return []
    found = CITATION_PATTERN.findall(text)
    if len(found) > 3:
        unique = sorted(set(found))
        return [f"No sources were retrieved but text contains {len(found)} inline citation(s) — verify content accuracy"]
    return []


def check_transition_overuse(text: str, max_per_paragraph: float = 2.0) -> list[str]:
    transition_words = ["however", "furthermore", "moreover", "therefore", "thus",
                         "consequently", "additionally", "nevertheless"]
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    issues = []
    for i, para in enumerate(paragraphs):
        # strip punctuation so "However," matches "however", not just "however" with nothing attached
        words = re.findall(r"\b[a-z]+\b", para.lower())
        count = sum(words.count(tw) for tw in transition_words)
        if count > max_per_paragraph:
            issues.append(f"Paragraph {i+1} uses {count} transition words — likely reads as formulaic")
    return issues


def check_repetition(text: str, previous_sections_summary: str, overlap_threshold: int = 5) -> list[str]:
    """
    Crude but effective: flags this section if it shares long word sequences
    with what earlier sections already covered (per _summarize_for_context).
    Using 5-grams and a higher count threshold to avoid false positives from
    shared topic vocabulary.
    """
    def ngrams(s, n=overlap_threshold):
        words = s.lower().split()
        return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}

    current_ngrams = ngrams(text)
    previous_ngrams = ngrams(previous_sections_summary)
    overlap = current_ngrams & previous_ngrams
    if len(overlap) > 10:
        return [f"Section repeats phrasing already used earlier ({len(overlap)} overlapping {overlap_threshold}-word sequences)"]
    return []


def check_style_drift(text: str, style_profile) -> list[str]:
    """Flags if this section's sentence rhythm is way off from the uploaded papers' profile."""
    sentences = [s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not sentences:
        return []
    lengths = [len(s.split()) for s in sentences]
    avg = statistics.mean(lengths)

    target = style_profile.avg_sentence_length
    if target and abs(avg - target) > target * 0.5:  # more than 50% off target
        return [f"Sentence length ({avg:.0f} words avg) diverges significantly from source style ({target:.0f} words avg)"]
    return []


def evaluate_section(text: str, sources_used: list[dict], style_profile, previous_sections_summary: str = "") -> EvaluationResult:
    issues = []
    issues += check_fabricated_citations(text, sources_used)
    issues += check_transition_overuse(text)
    issues += check_repetition(text, previous_sections_summary)
    issues += check_style_drift(text, style_profile)

    return EvaluationResult(passed=len(issues) == 0, issues=issues)