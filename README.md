# Research Style Writer (RSW)

A Generative AI system that produces human-like academic research papers with near-zero AI detection scores. It learns writing style from reference papers, grounds content in live web sources, and applies a multi-pass humanization pipeline to eliminate detectable AI patterns.

## Key Features

- **Style-Aware Generation** — Extracts 12 quantitative style features from reference papers and instructs the LLM to match them
- **Web-Grounded Content** — Every section is backed by real, current sources via Tavily API (no hallucinated citations)
- **5-Pass Humanization Pipeline** — Structural manipulation + LLM-based multi-candidate re-ranking + lexical processing + quality gate + final enforcement
- **Automated Quality Assurance** — 5 evaluation checks with iterative refinement before humanization
- **Academic Image Integration** — Relevant figures from Tavily, Unsplash, Wikimedia, and Europe PMC with cross-section deduplication
- **Professional Export** — Complete papers as PDF (with embedded images and DejaVu fonts) or DOCX
- **Multi-Provider LLM Support** — xAI Grok, Google Gemini, Groq, and OpenRouter

## How It Works

1. **Upload** 3-8 reference PDF papers for writing style extraction
2. **Enter** a research topic and keywords
3. **Generate** — the system searches the web, builds structured prompts, generates each section, evaluates quality, rewrites if needed, then humanizes
4. **Export** the finished paper as PDF or DOCX

### Humanization Pipeline

The core innovation targeting Turnitin-style detection signals:

| Pass | What It Does | Method |
|------|-------------|--------|
| 1 | Structural variation | Burstiness injection, sentence restructuring, paragraph mixing (pure Python) |
| 2 | LLM rewrite | 3 candidates with different personas, scored and best picked |
| 3 | Lexical processing | Perplexity-raising word swaps, synonym rotation, human artifacts |
| 4 | Quality gate | Composite >= 80, burstiness CV >= 0.45, variety >= 0.35 |
| 5 | Final enforcement | Clean up any uniformity the LLM re-introduced |

### Detection Metrics (Before vs After)

| Metric | Before | After |
|--------|--------|-------|
| Burstiness CV | 0.15 - 0.25 | > 0.45 |
| Residual AI Tells | 5 - 15 words | 0 |
| Sentence Variety | 0.15 - 0.25 | > 0.35 |
| Composite Score | 30 - 50 | > 80 |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| UI | Streamlit |
| LLM Orchestration | LangChain |
| LLM Providers | Google Gemini, xAI Grok, Groq, OpenRouter |
| Web Search | Tavily API |
| Embeddings | SentenceTransformer (all-MiniLM-L6-v2) |
| Vector Store | FAISS |
| PDF Extraction | PyMuPDF (fitz) with VLM fallback |
| Export | fpdf2 (PDF), python-docx (DOCX) |
| Image Sources | Tavily, Unsplash, Wikimedia, Europe PMC |

## Project Structure

```
RSW/
  apps/
    app_scenario1.py              # Streamlit app — Novice mode
    app_scenario2.py              # Streamlit app — Assisted mode
  src/
    core/
      config.py                   # Configuration & API keys
      ingestion/                  # PDF loader, chunker, VLM extractor
      style/                      # Style analysis (12 features)
      llm/                        # Multi-provider LLM factory
      retrieval/                  # Tavily web search
      illustration/               # Image search (4 sources)
      humanizer/                  # 5-pass humanization pipeline
        burstiness.py             # Sentence length variation
        sentence_restructurer.py  # Clause reordering
        paragraph_mixer.py        # Paragraph structure variation
        reranker.py               # 3-candidate scoring
        lexical_booster.py        # Perplexity word swaps
        human_artifacts.py        # Hedges, parentheticals
        humanizer.py              # Pipeline orchestrator
      export/                     # PDF & DOCX generators
      embeddings/                 # Embedding model wrapper
      vectorstore/                # FAISS vector store
    scenario1/                    # Novice mode (style refs + topic)
      generator.py                # Section & paper generation
      evaluator.py                # 5 quality checks
      prompts.py                  # LLM prompt construction
    scenario2/                    # Assisted mode (user materials + KB)
      ingestion.py                # Multi-format ingestion
      knowledge_base.py           # RAG knowledge base
      generator.py                # KB-aware generation
      prompts.py                  # KB-aware prompts
  deliverables/
    scenario1/
      rsw_scenario1_walkthrough.ipynb   # Jupyter notebook walkthrough
      rsw_scenario1_walkthrough.py      # Script version
      app_scenario1.py                  # Streamlit app copy
      RSW_Project_Report.pdf            # Project report
      generate_report.py                # Report generator script
```

## Setup

### 1. Clone and install

```bash
git clone https://github.com/Shantanu-0101/Research_Style_Writer.git
cd Research_Style_Writer
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file in the project root:

```env
# LLM Provider (choose one: xai, gemini, groq, openrouter)
LLM_PROVIDER=gemini

# Provider API keys (set the one matching your provider)
GOOGLE_API_KEY=your_key_here
XAI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here

# Web search (required for grounded content)
TAVILY_API_KEY=your_key_here

# Image search (optional)
UNSPLASH_API_KEY=your_key_here
```

### 3. Run

```bash
# Novice mode — upload reference PDFs, enter topic, generate
streamlit run apps/app_scenario1.py

# Assisted mode — upload your own research materials + reference papers
streamlit run apps/app_scenario2.py
```

## Usage Modes

### Novice Mode
For users starting from scratch. Upload 3-8 reference papers for style, enter a topic and keywords, and generate a complete paper grounded in web search.

### Assisted Mode
For users with existing research materials. Upload PDFs, DOCX, PPTX, TXT, or TEX files to build a knowledge base. The system fills gaps with web search and generates sections grounded in your materials.

## Deliverables

The `deliverables/` folder contains materials for course inclusion:

1. **Jupyter Notebook** (`rsw_scenario1_walkthrough.ipynb`) — 12-step interactive walkthrough of the entire pipeline
2. **Python Script** (`rsw_scenario1_walkthrough.py`) — Same walkthrough as a runnable script
3. **Streamlit App** (`app_scenario1.py`) — Copy of the web application
4. **Project Report** (`RSW_Project_Report.pdf`) — Business understanding, project understanding, architecture diagram, flowchart, final outcome

## License

This project is for academic and educational purposes.
