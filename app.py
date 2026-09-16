import streamlit as st
from src.config import settings
from src.ingestion.pdf_loader import extract_text_from_uploads
from src.ingestion.chunker import chunk_pages
from src.embeddings.embedder import Embedder
from src.vectorstore.faiss_store import FaissStore
from src.generation.generator import generate_section
from src.generation.generator import generate_full_paper
from src.style.style_analyzer import analyze_style



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
            with st.spinner("Chunking and embedding..."):
                chunks = chunk_pages(pages)
                embedder = Embedder()
                vectors = embedder.embed_documents([c["text"] for c in chunks])

                store = FaissStore()
                store.build(chunks, vectors)
                style_profile = analyze_style(chunks)

                st.session_state.faiss_store = store
                st.session_state.embedder = embedder
                st.session_state.style_profile = style_profile 

            st.success(f"Processed {len(uploaded_files) - len(skipped)} papers into {len(chunks)} chunks.")

# Generate a section
st.header("2. Generate a section")

if st.session_state.faiss_store is None:
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
                result = generate_section(
                    topic=topic,
                    section=section,
                    embedder=st.session_state.embedder,
                    faiss_store=st.session_state.faiss_store,
                    style_profile=st.session_state.style_profile,
                    extra_instructions=extra_instructions,
                )
                st.subheader(f"Generated {section}")
                st.write(result["text"])

                with st.expander("Sources used"):
                    for source, page in result["sources"]:
                        st.write(f"- {source}, page {page}")

                st.download_button(
                    "Download as text",
                    result["text"],
                    file_name=f"{section.lower().replace(' ', '_')}.txt",
                )
            except Exception as e:
                st.error(f"Generation failed: {e}")

    if st.button("Generate All Sections"):
        for section_name, result in generate_full_paper(topic, st.session_state.embedder, st.session_state.faiss_store,style_profile=st.session_state.style_profile,extra_instructions=extra_instructions):
            st.subheader(section_name)
            st.write(result["text"])
            with st.expander(f"Sources used ({section_name})"):
                for source, page in result["sources"]:
                    st.write(f"- {source}, page {page}")