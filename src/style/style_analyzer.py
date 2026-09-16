import re
import statistics
from dataclasses import dataclass
from collections import Counter


@dataclass
class StyleProfile:
    
    avg_sentence_length: float
    sentence_length_std: float
    lexical_diversity: float
    top_transition_words: list[tuple[str, int]]
    passive_voice_ratio: float

    def to_prompt_text(self) -> str:
        """Formats the profile as plain English for injection into the LLM prompt."""
        transitions = ", ".join(w for w, _ in self.top_transition_words) or "minimal, varied"
        return (
            f"Average sentence length: {self.avg_sentence_length:.0f} words "
            f"(typical variation: ±{self.sentence_length_std:.0f} words — "
            f"{'mix short and long sentences' if self.sentence_length_std > 8 else 'fairly uniform sentence length'}). "
            f"Vocabulary richness: {'high, varied wording' if self.lexical_diversity > 0.5 else 'moderate, some repeated terms'}. "
            f"Frequently used transition/hedging words: {transitions}. "
            f"Passive voice usage: {'frequent' if self.passive_voice_ratio > 0.3 else 'occasional'}."
        )


TRANSITION_WORDS = [
    "however", "furthermore", "moreover", "therefore", "thus", "consequently",
    "additionally", "nevertheless", "specifically", "notably", "importantly",
]

# Heuristic passive-voice detector: "is/was/were/been/being" + a word ending in -ed.
# Not linguistically perfect (misses irregular past participles like "written",
# catches false positives like "interested"), but good enough as a style signal.
PASSIVE_PATTERN = re.compile(r"\b(is|was|were|been|being|are)\s+\w+ed\b", re.IGNORECASE)


def analyze_style(chunks: list[dict]) -> StyleProfile:
    """
    Computes style statistics from the full set of chunks (all uploaded
    papers), not a retrieval-dependent subset. Call this once right after
    building the vector store, cache the result, reuse for every section.
    """
    full_text = " ".join(c["text"] for c in chunks)

    # Split into sentences on ., !, or ? followed by whitespace.
    sentences = [s for s in re.split(r'(?<=[.!?])\s+', full_text) if s.strip()]
    sentence_lengths = [len(s.split()) for s in sentences]

    avg_len = statistics.mean(sentence_lengths) if sentence_lengths else 0
    # stdev needs at least 2 data points, guard against a tiny sample
    len_std = statistics.stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0

    # Lexical diversity = unique words / total words (type-token ratio).
    # Simple and interpretable, though it drifts lower on longer texts —
    # fine here since we're comparing profile to profile, not absolute truth.
    words = re.findall(r"\b\w+\b", full_text.lower())
    lexical_diversity = len(set(words)) / len(words) if words else 0

    # Count occurrences of each known transition word, keep the ones actually used
    word_counts = Counter(words)
    found_transitions = [(w, word_counts[w]) for w in TRANSITION_WORDS if word_counts[w] > 0]
    found_transitions.sort(key=lambda x: -x[1])  # most frequent first

    passive_matches = len(PASSIVE_PATTERN.findall(full_text))
    passive_voice_ratio = passive_matches / len(sentences) if sentences else 0

    return StyleProfile(
        avg_sentence_length=avg_len,
        sentence_length_std=len_std,
        lexical_diversity=lexical_diversity,
        top_transition_words=found_transitions[:5],
        passive_voice_ratio=passive_voice_ratio,
    )