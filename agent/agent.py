"""LangChain agent with tool-calling capabilities."""

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI


@tool
def search(query: str) -> str:
    """Search for information. Use this when you need to look up facts or current events."""
    # Placeholder: replace with TavilySearch or another search tool when needed
    return f"Search results for: {query}"


def create_search_agent():
    """Create an agent with search capabilities."""
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return create_agent(
        model=model,
        tools=[search],
        system_prompt="You are a helpful assistant. Use the search tool when you need to look up information.",
    )
