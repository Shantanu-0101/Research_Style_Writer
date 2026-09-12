# AI Research Paper Generator

A Streamlit application that generates human-like academic text by learning the writing style from uploaded research papers.

## Features
- Upload multiple research papers (PDF)
- Create a knowledge base using RAG
- Generate academic sections (Introduction, Literature Review, Methodology, etc.) in the style of the uploaded papers

## Tech Stack
- Streamlit
- LangChain
- FAISS
- Sentence Transformers
- Groq / OpenAI / Gemini

## How to Run
1. Clone the repository
2. Create a virtual environment
3. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Add your API keys in .env file
5. streamlit run app.py