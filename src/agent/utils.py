def deduplicate_and_format_sources(
    search_response, max_tokens_per_source, include_raw_content=False
):
    """
    Takes search response from either Tavily or Bing API and formats them.
    Limits the content to approximately max_tokens_per_source.

    Args:
        search_response: Either:
            - A dict with a 'results' key containing a list of search results (Tavily format)
            - A list of dicts with 'url'/'link', 'content'/'snippet', 'title' keys (Bing format)
            - A list of dicts with 'results' containing either of the above

    Returns:
        str: Formatted string with deduplicated sources
    """
    # Convert input to list of results
    sources_list = []
    
    if isinstance(search_response, dict):
        # Tavily format
        sources_list = search_response.get("results", [])
    elif isinstance(search_response, list):
        for response in search_response:
            if isinstance(response, dict):
                if "results" in response:
                    # Nested results format
                    sources_list.extend(response["results"])
                else:
                    # Direct result format (Bing)
                    sources_list.append(response)
    else:
        raise ValueError(
            "Input must be either a dict with 'results' or a list of search results"
        )

    # Deduplicate by URL, handling both Tavily and Bing formats
    unique_sources = {}
    for source in sources_list:
        # Handle both Bing (link) and Tavily (url) formats
        url = source.get("url") or source.get("link")
        if not url:
            continue
            
        if url not in unique_sources:
            # Create standardized source format
            unique_sources[url] = {
                "url": url,
                "title": source.get("title", "Unknown Title"),
                "content": source.get("content") or source.get("snippet", ""),
                "raw_content": source.get("raw_content", "")
            }

    # Format output
    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {source['title']}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        
        if include_raw_content and source["raw_content"]:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            raw_content = source["raw_content"]
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
        else:
            formatted_text += "\n"

    return formatted_text.strip()


def format_all_notes(completed_notes: list[str]) -> str:
    """Format a list of notes into a string"""
    formatted_str = ""
    for idx, company_notes in enumerate(completed_notes, 1):
        formatted_str += f"""
{'='*60}
Note: {idx}:
{'='*60}
Notes from research:
{company_notes}"""
    return formatted_str