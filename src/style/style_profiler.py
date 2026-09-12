import re


def profile_style(chunks: list[dict], sample_size: int = 20) -> str:
    
    sample = [c["text"] for c in chunks[:sample_size]]
    all_text = " ".join(sample)

    sentences = re.split(r'(?<=[.!?])\s+', all_text)
    sentences = [s for s in sentences if s.strip()]
    avg_sentence_len = (
        sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
    )

    transition_words = ["however", "furthermore", "moreover", "therefore", "thus", "consequently"]
    found_transitions = [w for w in transition_words if w in all_text.lower()]

    return (
        f"Average sentence length: ~{avg_sentence_len:.0f} words. "
        f"Common transition words used: {', '.join(found_transitions) if found_transitions else 'varied, not formulaic'}. "
        f"Tone: formal, academic, third-person, minimal first-person pronouns."
    )