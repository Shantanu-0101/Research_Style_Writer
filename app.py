import streamlit as st
from src.config import settings
from src.ingestion.pdf_loader import extract_text_from_uploads
from src.ingestion.chunker import chunk_pages
from src.generation.generator import generate_full_paper
from src.style.style_analyzer import analyze_style
from src.generation.generator import generate_reviewed_section, generate_full_paper



st.set_page_config(page_title="Research Style Writer", layout="wide")
st.title("Research Style Writer")

try:
    settings.validate()
except ValueError as e:
    st.error(f"Configuration error: {e}")
    st.stop()

if "faiss_store" not in st.session_state:
    st.session_state.faiss_store = None
if "embedder" not in st.session_state:
    st.session_state.embedder = None

# Upload & process papers
st.header("1. Upload reference papers")
uploaded_files = st.file_uploader(
    "Upload 3-8 research papers (PDF)",
    type="pdf",
    accept_multiple_files=True,
)

if uploaded_files:
    if not (settings.min_files <= len(uploaded_files) <= settings.max_files):
        st.warning(
            f"Please upload between {settings.min_files} and {settings.max_files} PDFs "
            f"(currently {len(uploaded_files)})."
        )
    elif st.button("Process papers"):
        with st.spinner("Extracting text..."):
            pages, skipped = extract_text_from_uploads(uploaded_files)
            if skipped:
                st.warning(f"Couldn't read: {', '.join(skipped)}")

        if not pages:
            st.error("No text could be extracted from any uploaded file.")
        else:
            with st.spinner("Analyzing style..."):
                chunks = chunk_pages(pages)
                style_profile = analyze_style(chunks)
                st.session_state.style_profile = style_profile

            st.success(f"Processed {len(uploaded_files) - len(skipped)} papers, analyzed writing style.")

# Generate a section
st.header("2. Generate a section")

if "style_profile" not in st.session_state:
    st.info("Upload and process papers above before generating.")
else:
    topic = st.text_input("Paper topic / title")
    section = st.selectbox(
        "Section to generate",
        ["Introduction", "Literature Review", "Methodology", "Results", "Conclusion"],
    )
    extra_instructions = st.text_area("Additional instructions (optional)")

    if st.button("Generate", disabled=not topic):
        with st.spinner("Generating..."):
            try:
                result = generate_reviewed_section(
                        topic=topic,
                        section=section,
                        style_profile=st.session_state.style_profile,
                        extra_instructions=extra_instructions,
                    )
                st.subheader(f"Generated {section}")
                st.write(result["text"])

                if result["passed"]:
                    st.caption(f"✅ Passed review after {result['attempts']} attempt(s)")
                else:
                    st.caption(f"⚠️ {result['attempts']} attempts made, unresolved issues:")
                    for issue in result["remaining_issues"]:
                        st.caption(f"- {issue}")

                with st.expander("Sources used"):
                    for src in result["sources"]:
                        st.write(f"- [{src['title']}]({src['url']})")

                st.download_button(
                    "Download as text",
                    result["text"],
                    file_name=f"{section.lower().replace(' ', '_')}.txt",
                )
            except Exception as e:
                st.error(f"Generation failed: {e}")

    if st.button("Generate All Sections"):
        for section_name, result in generate_full_paper(topic,style_profile=st.session_state.style_profile,extra_instructions=extra_instructions):
            st.subheader(section_name)
            st.write(result["text"])
            with st.expander(f"Sources used ({section_name})"):
                for src in result["sources"]:
                    st.write(f"- [{src['title']}]({src['url']})")


            st.session_state.generated_sections.append((section_name, result["text"]))

            if result["passed"]:
                st.caption(f"✅ Passed review after {result['attempts']} attempt(s)")
            else:
                st.caption(f"⚠️ {result['attempts']} attempts made, unresolved issues:")
                for issue in result["remaining_issues"]:
                    st.caption(f"- {issue}")

    if "generated_sections" not in st.session_state:
                        st.session_state.generated_sections = []


    from src.export.pdf_export import generate_pdf

    if st.session_state.generated_sections:
        pdf_bytes = generate_pdf(topic, st.session_state.generated_sections)
        st.download_button(
            "Export as PDF",
            pdf_bytes,
            file_name=f"{topic.lower().replace(' ', '_')}.pdf",
            mime="application/pdf",
        )