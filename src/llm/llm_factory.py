from langchain_groq import ChatGroq
from src.config import settings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.7):

    provider = settings.llm_provider

    if settings.llm_provider == "groq":
        return ChatGroq(
            model="openai/gpt-oss-120b",
            groq_api_key=settings.groq_api_key,
            temperature=temperature,
        )

    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,  # e.g. "gemini-2.0-flash"
            google_api_key=settings.google_api_key,
            temperature=temperature,
        )

    # if provider == "ollama":
    #     # No API key needed — talks to your local Ollama server
    #     return ChatOllama(
    #         model=settings.ollama_model,  #"llama3.1:8b-instruct-q4_K_M"
    #         temperature=temperature,
    #     )

    if provider == "openrouter":
        # OpenRouter exposes an OpenAI-compatible endpoint
        return ChatOpenAI(
            model=settings.openrouter_model,
            openai_api_key=settings.openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=temperature,
        )
    
    raise ValueError(f"Unsupported LLM_PROVIDER '{provider}'")