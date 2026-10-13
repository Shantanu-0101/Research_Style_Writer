"""Quick import and sanity test for all refined modules."""
from src.core.style.style_analyzer import StyleProfile, analyze_style
from src.core.humanizer.sentence_restructurer import restructure_sentences, measure_sentence_initial_variety
from src.core.humanizer.paragraph_mixer import mix_paragraphs
from src.core.humanizer.burstiness import inject_burstiness, measure_burstiness, _PUNCHY_INSERTS, _PUNCHY_BY_CATEGORY
from src.core.humanizer.lexical_booster import boost_perplexity, enforce_lexical_diversity, ACADEMIC_RARE_INJECTIONS
from src.core.humanizer.human_artifacts import inject_human_artifacts, HEDGES, PARENTHETICALS, CROSS_REFERENCES, CITATION_CONNECTORS
from src.core.humanizer.reranker import score_candidate, PROMPT_CANDIDATE_C
from src.scenario1.generator import check_post_humanization_quality

print("Punchy inserts:", len(_PUNCHY_INSERTS), "(was 15)")
print("Punchy categories:", len(_PUNCHY_BY_CATEGORY))
print("Hedges:", len(HEDGES), "(was 10)")
print("Parentheticals:", len(PARENTHETICALS), "(was 15)")
print("Cross-references:", len(CROSS_REFERENCES), "(was 8)")
print("Citation connectors:", len(CITATION_CONNECTORS), "(was 7)")
print("Academic rare injections:", len(ACADEMIC_RARE_INJECTIONS))
print("Candidate C prompt loaded:", len(PROMPT_CANDIDATE_C) > 100)

sp = StyleProfile(
    avg_sentence_length=18, sentence_length_std=8, lexical_diversity=0.55,
    top_transition_words=[("however", 10)], passive_voice_ratio=0.2,
    avg_paper_word_count=3000, avg_paragraph_length=90, paragraph_length_std=35,
    sentence_initial_variety=0.52, questions_per_1000_words=1.2,
    first_person_ratio=0.03, avg_paragraphs_per_section=6,
)
prompt_text = sp.to_prompt_text()
print("StyleProfile prompt length:", len(prompt_text), "chars")
assert "first-person plural" in prompt_text
assert "paragraph" in prompt_text.lower()
assert "sentence-initial" in prompt_text.lower()
assert "rhetorical" in prompt_text.lower()
print("StyleProfile assertions passed")

test = "The model performs well, although some limitations exist. The results show improvement. The data confirms this trend."
result = restructure_sentences(test)
print("Sentence restructurer: OK (" + str(len(result)) + " chars)")

test_paras = "First paragraph with enough words to be processed by the mixer module successfully here now today.\n\nSecond paragraph with another set of words to ensure proper processing of the text works well here.\n\nThird paragraph continues the discussion of the topic with more detail and elaboration."
result = mix_paragraphs(test_paras)
print("Paragraph mixer: OK (" + str(len(result)) + " chars)")

test_text = "This is a test. But scale matters. The results show that the approach works well; yet caveats remain. Why does this matter? And that changes things."
score = score_candidate(test_text)
print("Score candidate: composite=" + str(round(score["composite_score"], 1)))

quality = check_post_humanization_quality(test_text)
print("Post-humanization check:", len(quality["metrics"]), "metrics")

print("\nAll imports and tests PASSED")
