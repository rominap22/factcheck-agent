"""
Debate Arena — MCP Server
Provides three real-world data-fetching tools for the fact-check agent.
"""

from fastmcp import FastMCP

mcp = FastMCP("DebateArenaServer")


@mcp.tool()
def search_wikipedia(query: str) -> str:
    """Search Wikipedia and return a page summary.
    Best for established facts, historical events, and encyclopedic knowledge."""
    import wikipedia

    try:
        page = wikipedia.page(query, auto_suggest=True)
        return (
            f"Title: {page.title}\n\n"
            f"Summary: {page.summary}\n\n"
            f"Source URL: {page.url}"
        )
    except wikipedia.DisambiguationError as e:
        options = ", ".join(e.options[:5])
        return (
            f"Search for '{query}' returned a disambiguation page. "
            f"Try a more specific term. Suggestions: {options}"
        )
    except wikipedia.PageError:
        return (
            f"No Wikipedia page found for '{query}'. "
            "Try a different or more specific search term."
        )
    except Exception as e:
        return f"Wikipedia search failed: {e}"


@mcp.tool()
def search_current_web(query: str) -> str:
    """Search the live web via DuckDuckGo and return the top 3 results.
    Best for recent events, news, and information not in encyclopedias."""
    from ddgs import DDGS

    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return f"No web results found for '{query}'."

        sections = []
        for i, r in enumerate(results, 1):
            sections.append(
                f"Result {i}:\n"
                f"  Title: {r['title']}\n"
                f"  Snippet: {r['body']}\n"
                f"  URL: {r['href']}"
            )
        return "\n\n".join(sections)
    except Exception as e:
        return f"Web search failed: {e}"


@mcp.tool()
def search_arxiv_papers(query: str) -> str:
    """Search ArXiv for the top 2 academic papers matching the query.
    Best for scientific, technical, or research-backed claims."""
    import arxiv

    try:
        search = arxiv.Search(
            query=query,
            max_results=2,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        client = arxiv.Client()
        results = list(client.results(search))

        if not results:
            return f"No ArXiv papers found for '{query}'."

        sections = []
        for i, paper in enumerate(results, 1):
            authors = ", ".join(a.name for a in paper.authors[:3])
            if len(paper.authors) > 3:
                authors += " et al."
            abstract = paper.summary[:600]
            if len(paper.summary) > 600:
                abstract += "…"
            sections.append(
                f"Paper {i}:\n"
                f"  Title: {paper.title}\n"
                f"  Authors: {authors}\n"
                f"  Abstract: {abstract}\n"
                f"  URL: {paper.entry_id}"
            )
        return "\n\n".join(sections)
    except Exception as e:
        return f"ArXiv search failed: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
