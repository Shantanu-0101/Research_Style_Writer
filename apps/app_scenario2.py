"""
Scenario 2 — Assisted: AI Research Paper Generator with User Materials

User uploads multi-format research materials (PDFs, DOCX, PPTX, TXT, TEX)
and optionally provides web URLs. The system ingests everything into a
knowledge base, fills gaps with web search, and generates humanized sections.
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re

from src.core.config import settings
from src.scenario2.ingestion import extract_from_uploads, fetch_url_content, SUPPORTED_EXTENSIONS
from src.core.ingestion.chunker import chunk_pages
from src.core.style.style_analyzer import analyze_style
from src.scenario2.knowledge_base import KnowledgeBase
from src.scenario2.generator import generate_full_paper, generate_reviewed_section, check_post_humanization_quality
from src.core.export.pdf_export import generate_pdf
from src.core.export.docx_export import generate_docx
from src.core.humanizer.humanizer import humanize_section
from src.core.detection.ai_detector import check_ai_score


def render_section_content(text: str):
    image_pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")
    lines = text.split("\n")
    for line in lines:
        match = image_pattern.search(line)
        if match:
            caption = match.group(1)
            img_path = match.group(2).strip()
            if os.path.exists(img_path):
                st.image(img_path, caption=caption, use_container_width=True)
            else:
                st.warning(f"Image not found: {img_path}")
        else:
            st.markdown(line)


def render_ai_score(text: str, zerogpt_key: str, label: str = ""):
    with st.spinner(f"Checking AI detection score{' for ' + label if label else ''}..."):
        result = check_ai_score(text, api_key=zerogpt_key)

    if result["error"]:
        st.warning(f"AI Check failed: {result['error']}")
        return

    pct = result["ai_percentage"]
    lbl = result["label"]
    if lbl == "Human":
        st.success(f"AI Detection: **{pct}%** — {result['feedback']}")
    elif lbl == "Mixed":
        st.warning(f"AI Detection: **{pct}%** — {result['feedback']}")
    elif lbl == "Too Short":
        st.info(result["feedback"])
    else:
        st.error(f"AI Detection: **{pct}%** — {result['feedback']}")


def main():
    st.set_page_config(page_title="Research Style Writer — Scenario 2", layout="wide")
    st.title("Research Style Writer")
    st.caption("Scenario 2: Assisted — Upload your research materials, build a knowledge base, generate academic sections")

    try:
        settings.validate()
    except ValueError as e:
        st.error(f"Configuration error: {e}")
        st.stop()

    # Session state
    if "s2_style_profile" not in st.session_state:
        st.session_state.s2_style_profile = None
    if "s2_kb" not in st.session_state:
        st.session_state.s2_kb = None
    if "s2_kb_stats" not in st.session_state:
        st.session_state.s2_kb_stats = None
    if "s2_generated_sections" not in st.session_state:
        st.session_state.s2_generated_sections = []

    # ── Sidebar ──
    with st.sidebar:
        st.header("Options")
        use_humanizer = st.toggle("Humanize output", value=True)
        humanizer_strength = 2
        if use_humanizer:
            humanizer_strength = st.select_slider(
                "Humanization Strength",
                options=[1, 2, 3],
                value=2,
                format_func=lambda x: {1: "Standard (1 pass)", 2: "Deep (2 passes)", 3: "Maximum (3 passes)"}[x],
            )
            st.info(f"Pipeline: Structural variation + {humanizer_strength}x LLM rewrite + Lexical processing + Quality gate")

        st.divider()
        st.subheader("AI Detection")
        _default_zgkey = settings.zerogpt_api_key or ""
        zerogpt_key = st.text_input("ZeroGPT API Key", type="password", value=_default_zgkey)
        if _default_zgkey:
            st.caption("Key loaded from .env")

    # ── Step 1: Upload research materials ──
    st.header("1. Upload research materials")
    st.markdown(f"Supported formats: **{', '.join(sorted(SUPPORTED_EXTENSIONS))}**")

    col_files, col_urls = st.columns([2, 1])

    with col_files:
        uploaded_files = st.file_uploader(
            "Upload your research papers, notes, slides, LaTeX sources",
            type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
            accept_multiple_files=True,
        )

    with col_urls:
        urls_text = st.text_area(
            "Web URLs (one per line, optional)",
            placeholder="https://arxiv.org/abs/...\nhttps://blog.example.com/...",
            help="Paste URLs to blogs, arXiv pages, or other web resources to include in your knowledge base.",
        )

    # ── Step 2: Upload style reference papers ──
    st.header("2. Upload style reference papers")
    st.caption("Upload 3-8 PDF research papers whose writing style you want to match (separate from your research materials above)")

    style_files = st.file_uploader(
        "Style reference papers (PDF only)",
        type="pdf",
        accept_multiple_files=True,
        key="style_uploader",
    )

    # ── Build Knowledge Base ──
    if uploaded_files or urls_text.strip():
        if st.button("Build Knowledge Base & Analyze Style"):
            # Process uploaded files
            all_pages = []
            skipped = []

            if uploaded_files:
                with st.spinner(f"Extracting text from {len(uploaded_files)} file(s)..."):
                    pages, skip = extract_from_uploads(uploaded_files)
                    all_pages.extend(pages)
                    skipped.extend(skip)
                    if skip:
                        st.warning(f"Could not read: {', '.join(skip)}")

            # Process URLs
            if urls_text.strip():
                urls = [u.strip() for u in urls_text.strip().split("\n") if u.strip()]
                with st.spinner(f"Fetching {len(urls)} URL(s)..."):
                    for url in urls:
                        url_pages = fetch_url_content(url)
                        if url_pages:
                            all_pages.extend(url_pages)
                        else:
                            skipped.append(url)
                            st.warning(f"Could not fetch: {url}")

            if not all_pages:
                st.error("No text could be extracted from any source.")
            else:
                # Build knowledge base
                with st.spinner("Building knowledge base (embedding + indexing)..."):
                    kb = KnowledgeBase()
                    kb_stats = kb.build(all_pages)
                    st.session_state.s2_kb = kb
                    st.session_state.s2_kb_stats = kb_stats

                st.success(f"Knowledge base built: **{kb_stats['total_chunks']}** chunks from **{len(kb_stats['sources'])}** source(s)")

                with st.expander("Knowledge Base Sources"):
                    for src in kb_stats["sources"]:
                        st.write(f"- {src}")

            # Style analysis (from style reference papers OR from the uploaded materials)
            style_pages = []
            if style_files:
                if len(style_files) < settings.min_files:
                    st.warning(f"Upload at least {settings.min_files} style reference PDFs for reliable style analysis.")
                else:
                    with st.spinner("Analyzing writing style from reference papers..."):
                        from src.core.ingestion.pdf_loader import extract_text_from_uploads as extract_pdf_uploads
                        style_pages_raw, style_skipped = extract_pdf_uploads(style_files)
                        style_pages = style_pages_raw
                        if style_skipped:
                            st.warning(f"Style papers skipped: {', '.join(style_skipped)}")

            if not style_pages and all_pages:
                st.info("No separate style papers — using uploaded materials for style analysis.")
                # Filter to only PDF sources for style (best quality)
                pdf_pages = [p for p in all_pages if p["source"].lower().endswith(".pdf")]
                style_pages = pdf_pages if pdf_pages else all_pages

            if style_pages:
                with st.spinner("Analyzing style..."):
                    chunks = chunk_pages(style_pages)
                    style_profile = analyze_style(chunks)
                    st.session_state.s2_style_profile = style_profile
                st.success("Writing style analyzed.")

    # ── Show KB status ──
    if st.session_state.s2_kb_stats:
        stats = st.session_state.s2_kb_stats
        st.sidebar.divider()
        st.sidebar.subheader("Knowledge Base")
        st.sidebar.metric("Chunks Indexed", stats["total_chunks"])
        st.sidebar.metric("Sources", len(stats["sources"]))

    # ── Step 3: Generate ──
    st.header("3. Generate paper sections")

    if st.session_state.s2_style_profile is None or st.session_state.s2_kb is None:
        st.info("Upload materials and build the knowledge base first.")
    else:
        topic = st.text_input("Paper topic / title", key="s2_topic")
        keywords = st.text_area(
            "Keywords (comma-separated)",
            placeholder="e.g., Knowledge Graph, Meta-path, LLM, Recommendation",
            key="s2_keywords",
        )
        section = st.selectbox(
            "Section to generate",
            ["Abstract", "Introduction", "Literature Review", "Methodology", "Results", "Conclusion", "References"],
            key="s2_section",
        )
        extra_instructions = st.text_area("Additional instructions (optional)", key="s2_extra")
        include_visuals = st.checkbox("Include real academic images", value=True, key="s2_visuals")

        kb = st.session_state.s2_kb

        if st.button("Generate Section", disabled=not topic, key="s2_gen_single"):
            # Show coverage analysis
            coverage = kb.assess_coverage(topic, [section], keywords)
            sec_coverage = coverage.get(section, {})
            if sec_coverage.get("has_gap"):
                st.info(f"Your materials have limited coverage for {section} ({sec_coverage.get('total_words', 0)} words found). Web search will supplement.")
            else:
                st.success(f"Good coverage for {section}: {sec_coverage.get('total_words', 0)} words from your materials.")

            with st.spinner(f"Generating {section}..."):
                try:
                    result = generate_reviewed_section(
                        topic=topic,
                        section=section,
                        style_profile=st.session_state.s2_style_profile,
                        kb=kb,
                        keywords=keywords,
                        extra_instructions=extra_instructions,
                        include_visuals=include_visuals,
                    )

                    if use_humanizer and section.lower() != "references":
                        with st.spinner(f"Humanizing ({humanizer_strength} pass{'es' if humanizer_strength > 1 else ''})..."):
                            result = humanize_section(
                                result,
                                style_notes=st.session_state.s2_style_profile.to_prompt_text(),
                                max_passes=humanizer_strength,
                            )

                    st.subheader(f"Generated {section}")
                    render_section_content(result["text"])

                    # Status
                    col_status, col_kb = st.columns(2)
                    with col_status:
                        if result["passed"]:
                            st.caption(f"Passed review after {result['attempts']} attempt(s)")
                        else:
                            st.caption(f"{result['attempts']} attempts, unresolved issues:")
                            for issue in result["remaining_issues"]:
                                st.caption(f"- {issue}")
                    with col_kb:
                        st.caption(f"KB chunks used: {result.get('kb_chunks_used', 0)}")
                        if result.get("has_gap"):
                            st.caption("Gap detected — supplemented with web search")

                    # Humanization metrics
                    if use_humanizer and "humanization_metrics" in result:
                        metrics = result["humanization_metrics"]
                        with st.expander("Humanization Metrics"):
                            mc1, mc2, mc3, mc4 = st.columns(4)
                            mc1.metric("Burstiness CV", f"{metrics.get('burstiness_cv', 0):.3f}")
                            mc2.metric("AI Tells", metrics.get("residual_tells", 0))
                            mc3.metric("Variety", f"{metrics.get('sentence_initial_variety', 0):.3f}")
                            mc4.metric("Composite", f"{metrics.get('composite_score', 0):.0f}")

                        quality = check_post_humanization_quality(result["text"])
                        for warning in quality.get("warnings", []):
                            st.warning(warning)

                    with st.expander("Sources used"):
                        for src in result["sources"]:
                            st.write(f"- [{src['title']}]({src['url']})")

                    col_dl, col_ai = st.columns([2, 1])
                    with col_dl:
                        st.download_button("Download as text", result["text"], file_name=f"{section.lower().replace(' ', '_')}.txt")
                    with col_ai:
                        if st.button("Check AI Score", key=f"s2_ai_single_{section}"):
                            render_ai_score(result["text"], zerogpt_key, label=section)

                except Exception as e:
                    st.error(f"Generation failed: {e}")

        if st.button("Generate All Sections", key="s2_gen_all"):
            st.session_state.s2_generated_sections = []

            # Coverage overview
            all_section_names = ["Abstract", "Introduction", "Literature Review", "Methodology", "Results", "Conclusion"]
            coverage = kb.assess_coverage(topic, all_section_names, keywords)
            with st.expander("Knowledge Base Coverage"):
                for sec_name, cov in coverage.items():
                    if cov.get("has_gap"):
                        st.warning(f"{sec_name}: Limited coverage ({cov.get('total_words', 0)} words) — will supplement with web search")
                    else:
                        st.success(f"{sec_name}: Good coverage ({cov.get('total_words', 0)} words)")

            with st.spinner("Generating full paper..."):
                for section_name, result in generate_full_paper(
                    topic=topic,
                    style_profile=st.session_state.s2_style_profile,
                    kb=kb,
                    keywords=keywords,
                    extra_instructions=extra_instructions,
                    include_visuals=include_visuals,
                ):
                    if use_humanizer and section_name.lower() != "references":
                        with st.spinner(f"Humanizing {section_name}..."):
                            result = humanize_section(
                                result,
                                style_notes=st.session_state.s2_style_profile.to_prompt_text(),
                                max_passes=humanizer_strength,
                            )

                    st.subheader(section_name)
                    render_section_content(result["text"])

                    with st.expander(f"Sources ({section_name})"):
                        for src in result["sources"]:
                            st.write(f"- [{src['title']}]({src['url']})")

                    st.session_state.s2_generated_sections.append((section_name, result["text"]))

                    # Per-section metrics
                    col_info, col_ai = st.columns([3, 1])
                    with col_info:
                        status_parts = []
                        if result["passed"]:
                            status_parts.append(f"Passed ({result['attempts']} attempt(s))")
                        else:
                            status_parts.append(f"{result['attempts']} attempts")
                        status_parts.append(f"KB: {result.get('kb_chunks_used', 0)} chunks")
                        if result.get("has_gap"):
                            status_parts.append("gap filled by web")
                        st.caption(" | ".join(status_parts))

                        if use_humanizer and "humanization_metrics" in result:
                            m = result["humanization_metrics"]
                            st.caption(f"CV: {m.get('burstiness_cv', 0):.3f} | Tells: {m.get('residual_tells', 0)} | Variety: {m.get('sentence_initial_variety', 0):.3f}")

                    with col_ai:
                        if st.button("Check AI Score", key=f"s2_ai_{section_name}"):
                            render_ai_score(result["text"], zerogpt_key, label=section_name)

        # Export
        if st.session_state.s2_generated_sections:
            st.divider()
            st.subheader("Export Options")
            col1, col2 = st.columns(2)
            with col1:
                pdf_bytes = generate_pdf(topic, st.session_state.s2_generated_sections)
                st.download_button("Export as PDF", pdf_bytes, file_name=f"{topic.lower().replace(' ', '_')}.pdf", mime="application/pdf", use_container_width=True)
            with col2:
                docx_bytes = generate_docx(topic, st.session_state.s2_generated_sections)
                st.download_button("Export as Word (DOCX)", docx_bytes, file_name=f"{topic.lower().replace(' ', '_')}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)


if __name__ == "__main__":
    main()
