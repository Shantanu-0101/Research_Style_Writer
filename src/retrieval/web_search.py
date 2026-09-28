from langchain_community.tools.tavily_search import TavilySearchResults
from src.config import settings


def search_topic_content(topic: str, subtopic:  str, section: str, max_results: int = 5) -> list[dict]:
    """
    Searches the web for real, current information about the topic/section,
    to ground generated content in actual facts — separate from the uploaded
    papers, which are used ONLY for style now, not factual content.
    """
    search = TavilySearchResults(api_key=settings.tavily_api_key, max_results=max_results)
    query = f"{topic} {subtopic} {section} research academic".strip()

    try:
        results = search.invoke(query)
    except Exception:
        return []  # fail gracefully — generation can proceed without web grounding

    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in results
    ]