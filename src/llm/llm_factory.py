from langchain_groq import ChatGroq
from src.config import settings


def get_llm(temperature: float = 0.7):
    """
    Returns a LangChain chat model based on settings.llm_provider.
    Only 'groq' is implemented
    """
    if settings.llm_provider == "groq":
        return ChatGroq(
            model="openai/gpt-oss-120b",
            groq_api_key=settings.groq_api_key,
            temperature=temperature,
        )
    raise ValueError(f"Unsupported LLM_PROVIDER '{settings.llm_provider}'")