"""
Scenario 2 generator — RAG-augmented paper generation.

Flow per section:
  1. Query knowledge base for relevant user-provided context
  2. Check coverage — if gap detected, supplement with web search
  3. Generate section using hybrid KB + web prompt
  4. Evaluate and rewrite if needed
  5. Images (reuses Scenario 1 image pipeline)
"""

import re
import os
import hashlib
import json
from pathlib import Path

from src.scenario2.prompts import build_generation_prompt, build_rewrite_prompt
from src.scenario2.knowledge_base import KnowledgeBase
from src.core.style.style_analyzer import StyleProfile
from src.core.llm.llm_factory import get_llm
from src.scenario1.evaluator import evaluate_section
from src.core.retrieval.web_search import search_topic_content
from src.core.illustration.image_search import get_academic_illustration
from src.core.humanizer.burstiness import measure_burstiness, _split_into_sentences
from src.core.humanizer.reranker import score_candidate
from src.core.humanizer.sentence_restructurer import measure_sentence_initial_variety

DEFAULT_SECTIONS = ["Abstract", "Introduction", "Literature Review", "Methodology", "Results", "Conclusion", "References"]

IMAGE_CACHE_DIR = Path("data/image_cache")
IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_SECTION_SEARCH_MODIFIERS = {
    "introduction": "overview concept diagram",
    "literature review": "survey comparison taxonomy",
    "methodology": "framework architecture pipeline",
    "results": "results analysis performance chart",
    "conclusion": "summary implications future directions",
}


def _get_topic_cache_key(topic: str, keywords: str = "") -> str:
    combined = f"s2|{topic.lower().strip()}|{keywords.lower().strip()}"
    return hashlib.md5(combined.encode()).hexdigest()[:12]


def _load_topic_image_cache(topic: str, keywords: str = "") -> dict:
    cache_file = IMAGE_CACHE_DIR / f"{_get_topic_cache_key(topic, keywords)}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}


def _save_topic_image_cache(topic: str, keywords: str = "", cache: dict = None):
    if cache is None:
        return
    cache_file = IMAGE_CACHE_DIR / f"{_get_topic_cache_key(topic, keywords)}.json"
    try:
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=2)
    except:
        pass


def _build_image_query(topic: str, section: str, keywords: str = "", caption_hint: str = "") -> str:
    kw_list = [k.strip() for k in keywords.split(",") if k.strip()][:3]
    kw_part = " ".join(kw_list)
    modifier = _SECTION_SEARCH_MODIFIERS.get(section.lower(), "")

    if caption_hint and len(caption_hint) > 10:
        return f"{caption_hint} {kw_part}".strip()

    parts = [topic]
    if kw_part:
        parts.append(kw_part)
    if modifier:
        parts.append(modifier)
    return " ".join(parts)


def _process_visual_placeholders(text: str, topic: str, section: str, keywords: str = "", topic_image_cache: dict = None, used_image_paths: set = None) -> tuple[str, list[dict]]:
    if topic_image_cache is None:
        topic_image_cache = {}
    if used_image_paths is None:
        used_image_paths = set()

    pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")
    image_sources = []
    section_key = section.lower()

    def replace_match(match):
        caption = match.group(1)
        spec = match.group(2)

        if section_key in topic_image_cache:
            cached_path = topic_image_cache[section_key]
            if os.path.exists(cached_path) and cached_path not in used_image_paths:
                used_image_paths.add(cached_path)
                image_sources.append({"title": "Cached Image", "url": "#"})
                return f"![{caption}]({cached_path})"

        caption_hint = ""
        if "image_query:" in spec:
            caption_hint = spec.replace("image_query:", "").strip(' "\'')
        elif spec.strip():
            caption_hint = spec.strip(' "\'')

        query = _build_image_query(topic, section, keywords, caption_hint)
        img_data = get_academic_illustration(query, used_paths=used_image_paths)
        if img_data and os.path.exists(img_data["local_path"]):
            topic_image_cache[section_key] = img_data["local_path"]
            used_image_paths.add(img_data["local_path"])
            image_sources.append({
                "title": f"Image: {img_data.get('source_name', 'Web')}",
                "url": img_data.get("source_url", "#"),
            })
            return f"![{caption}]({img_data['local_path']})"
        return ""

    processed_text = pattern.sub(replace_match, text)

    if section.lower() not in ["references", "abstract"] and not any("![" in line for line in processed_text.split('\n') if line.strip()):
        auto_caption = f"{section} — {topic}"
        img_path = None
        img_source_entry = None

        if section_key in topic_image_cache and os.path.exists(topic_image_cache[section_key]) and topic_image_cache[section_key] not in used_image_paths:
            img_path = topic_image_cache[section_key]
            used_image_paths.add(img_path)
            img_source_entry = {"title": "Cached Image", "url": "#"}
        else:
            fallback_query = _build_image_query(topic, section, keywords)
            img_data = get_academic_illustration(fallback_query, used_paths=used_image_paths)
            if img_data and os.path.exists(img_data["local_path"]):
                img_path = img_data["local_path"]
                topic_image_cache[section_key] = img_path
                used_image_paths.add(img_path)
                img_source_entry = {
                    "title": f"Image: {img_data.get('source_name', 'Web')}",
                    "url": img_data.get("source_url", "#"),
                }

        if img_path:
            tag = f"![{auto_caption}]({img_path})"
            paragraphs = [p.strip() for p in processed_text.split("\n\n") if p.strip()]
            if len(paragraphs) >= 3:
                paragraphs.insert(2, tag)
            else:
                paragraphs.append(tag)
            processed_text = "\n\n".join(paragraphs)
            if img_source_entry:
                image_sources.append(img_source_entry)

    return processed_text, image_sources


def _generate_fallback_section(topic: str, section: str, kb_context: list[dict], web_results: list[dict]) -> str:
    parts = [f"This section addresses {section.lower()} aspects of {topic}."]

    for c in kb_context[:2]:
        if c.get("text"):
            parts.append(f"From the user's materials ({c['source']}): {c['text'][:300]}")

    for r in web_results[:2]:
        if r.get("content"):
            parts.append(f"According to {r.get('title', 'a source')}, {r['content'][:300]}")

    parts.append(f"Further analysis is needed for a comprehensive treatment of {section.lower()} in {topic}.")
    return "\n\n".join(parts)


def generate_section(
    topic: str,
    section: str,
    style_profile: StyleProfile,
    kb: KnowledgeBase,
    keywords: str = "",
    extra_instructions: str = "",
    previous_sections_summary: str = "",
    include_visuals: bool = True,
    topic_image_cache: dict = None,
    used_image_paths: set = None,
) -> dict:
    # 1. Retrieve from knowledge base
    kb_context = kb.retrieve_for_section(topic, section, keywords)
    total_kb_words = sum(len(c["text"].split()) for c in kb_context)
    has_gap = total_kb_words < 200

    # 2. Web search — always run for supplementary context, but flag gap status
    web_results = search_topic_content(topic, "", section)

    # 3. Generate
    prompt = build_generation_prompt(
        topic=topic,
        section=section,
        kb_context=kb_context,
        web_results=web_results,
        style_notes=style_profile.to_prompt_text(),
        style_profile=style_profile,
        keywords=keywords,
        extra_instructions=extra_instructions,
        previous_sections_summary=previous_sections_summary,
        include_visuals=include_visuals,
        has_gap=has_gap,
    )

    llm = get_llm()
    generated_text = ""
    for retry in range(3):
        try:
            response = llm.invoke(prompt)
            text = response.content if response.content else ""
            if text and len(text.strip()) > 80:
                generated_text = text
                break
            print(f"Section '{section}' short text, retry {retry + 1}/3")
        except Exception as e:
            print(f"Section '{section}' error: {e}, retry {retry + 1}/3")

    if not generated_text or len(generated_text.strip()) < 80:
        generated_text = _generate_fallback_section(topic, section, kb_context, web_results)

    sources_used = [{"title": r["title"], "url": r["url"]} for r in web_results]
    # Add KB source references
    kb_sources = list({c["source"] for c in kb_context})
    for src in kb_sources:
        sources_used.append({"title": f"User material: {src}", "url": "#"})

    if include_visuals and section.lower() != "references":
        generated_text, img_sources = _process_visual_placeholders(
            generated_text, topic, section, keywords=keywords,
            topic_image_cache=topic_image_cache, used_image_paths=used_image_paths,
        )
        sources_used.extend(img_sources)

    return {
        "text": generated_text,
        "sources": sources_used,
        "kb_chunks_used": len(kb_context),
        "has_gap": has_gap,
    }


def generate_reviewed_section(
    topic: str,
    section: str,
    style_profile: StyleProfile,
    kb: KnowledgeBase,
    keywords: str = "",
    extra_instructions: str = "",
    previous_sections_summary: str = "",
    max_attempts: int = 3,
    include_visuals: bool = True,
    topic_image_cache: dict = None,
    used_image_paths: set = None,
) -> dict:
    result = generate_section(
        topic=topic,
        section=section,
        style_profile=style_profile,
        kb=kb,
        keywords=keywords,
        extra_instructions=extra_instructions,
        previous_sections_summary=previous_sections_summary,
        include_visuals=include_visuals,
        topic_image_cache=topic_image_cache,
        used_image_paths=used_image_paths,
    )

    attempt = 1
    evaluation = evaluate_section(
        result["text"], result["sources"], style_profile,
        previous_sections_summary, section_name=section,
    )

    while not evaluation.passed and attempt < max_attempts:
        result["text"] = _rewrite_section(
            topic=topic,
            section=section,
            original_text=result["text"],
            issues=evaluation.issues,
            style_profile=style_profile,
        )
        attempt += 1
        evaluation = evaluate_section(
            result["text"], result["sources"], style_profile,
            previous_sections_summary, section_name=section,
        )

    return {
        "text": result["text"],
        "sources": result["sources"],
        "attempts": attempt,
        "passed": evaluation.passed,
        "remaining_issues": evaluation.issues,
        "kb_chunks_used": result.get("kb_chunks_used", 0),
        "has_gap": result.get("has_gap", False),
    }


def check_post_humanization_quality(text: str) -> dict:
    clean = re.compile(r"!\[(.*?)\]\((.*?)\)").sub("", text).strip()
    if not clean:
        return {"warnings": [], "metrics": {}}

    score = score_candidate(clean)
    variety = measure_sentence_initial_variety(clean)

    warnings = []
    if score["burstiness_cv"] < 0.45:
        warnings.append(f"Burstiness CV is low ({score['burstiness_cv']:.2f})")
    if score["residual_tells"] > 0:
        warnings.append(f"{score['residual_tells']} AI-tell word(s) remain")
    if variety < 0.35:
        warnings.append(f"Sentence-initial variety is low ({variety:.2f})")

    return {
        "warnings": warnings,
        "metrics": {
            "burstiness_cv": round(score["burstiness_cv"], 3),
            "residual_tells": score["residual_tells"],
            "composite_score": round(score["composite_score"], 1),
            "sentence_initial_variety": round(variety, 3),
        },
    }


def _rewrite_section(topic, section, original_text, issues, style_profile):
    image_pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")
    images = image_pattern.findall(original_text)
    clean_text = image_pattern.sub("", original_text).strip()

    prompt = build_rewrite_prompt(
        topic=topic,
        section=section,
        original_text=clean_text,
        issues=issues,
        style_notes=style_profile.to_prompt_text(),
    )
    llm = get_llm()
    response = llm.invoke(prompt)
    rewritten_text = response.content if (response and response.content) else clean_text

    if images:
        paragraphs = [p.strip() for p in rewritten_text.split("\n\n") if p.strip()]
        for caption, path_or_spec in images:
            tag = f"![{caption}]({path_or_spec})"
            if len(paragraphs) >= 3:
                paragraphs.insert(2, tag)
            else:
                paragraphs.append(tag)
        rewritten_text = "\n\n".join(paragraphs)

    return rewritten_text


def _summarize_for_context(section_name: str, text: str, max_words: int = 60) -> str:
    words = text.split()
    snippet = " ".join(words[:max_words])
    return f"- {section_name}: {snippet}{'...' if len(words) > max_words else ''}"


def generate_full_paper(
    topic: str,
    style_profile: StyleProfile,
    kb: KnowledgeBase,
    sections=None,
    keywords: str = "",
    extra_instructions: str = "",
    max_attempts: int = 3,
    include_visuals: bool = True,
):
    sections = sections or DEFAULT_SECTIONS
    running_summary_parts = []
    used_urls = set()
    used_image_paths = set()
    all_sources: list[dict] = []
    topic_image_cache = _load_topic_image_cache(topic, keywords)

    for section in sections:
        if section.lower() == "references":
            ref_text = "References\n\n"
            seen_urls = set()
            unique_sources = []
            for src in all_sources:
                url = src.get("url", "")
                if url and url != "#" and url not in seen_urls:
                    seen_urls.add(url)
                    unique_sources.append(src)

            if not unique_sources:
                ref_text += "No external web sources were cited in this paper."
            else:
                for i, src in enumerate(unique_sources, 1):
                    ref_text += f"{i}. {src.get('title', 'Source')}. Retrieved from {src.get('url', '')}\n\n"

            result = {
                "text": ref_text,
                "sources": [],
                "attempts": 1,
                "passed": True,
                "remaining_issues": [],
                "kb_chunks_used": 0,
                "has_gap": False,
            }
            yield section, result
            continue

        result = generate_reviewed_section(
            topic=topic,
            section=section,
            style_profile=style_profile,
            kb=kb,
            keywords=keywords,
            extra_instructions=extra_instructions,
            previous_sections_summary="\n".join(running_summary_parts),
            max_attempts=max_attempts,
            include_visuals=include_visuals,
            topic_image_cache=topic_image_cache,
            used_image_paths=used_image_paths,
        )

        for src in result.get("sources", []):
            url = src.get("url", "")
            if url and url != "#" and url not in used_urls:
                used_urls.add(url)
                all_sources.append(src)

        running_summary_parts.append(_summarize_for_context(section, result["text"]))
        yield section, result

    _save_topic_image_cache(topic, keywords, topic_image_cache)
