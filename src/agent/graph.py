import asyncio
import json

from tavily import AsyncTavilyClient
from langchain_anthropic import ChatAnthropic
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph

from pydantic import BaseModel, Field
from typing_extensions import Any, Literal

from agent.configuration import Configuration
from agent.state import InputState, OutputState, OverallState
from agent.utils import deduplicate_and_format_sources, format_all_notes
from agent.prompts import (
    EXTRACTION_PROMPT,
    REFLECTION_PROMPT,
    INFO_PROMPT,
    QUERY_WRITER_PROMPT,
)

# LLMs

rate_limiter = InMemoryRateLimiter(
    requests_per_second=4,
    check_every_n_seconds=0.1,
    max_bucket_size=10,  # Controls the maximum burst size.
)
claude_3_5_sonnet = ChatAnthropic(
    model="claude-3-5-sonnet-latest", temperature=0, rate_limiter=rate_limiter
)

# Search

tavily_async_client = AsyncTavilyClient()


class Queries(BaseModel):
    queries: list[str] = Field(
        description="List of search queries.",
    )


class ReflectionOutput(BaseModel):
    is_satisfactory: bool = Field(
        description="True if all required fields are well populated, False otherwise"
    )
    missing_fields: list[str] = Field(
        description="List of field names that are missing or incomplete"
    )
    reflection_search_queries: list[str] = Field(
        description="If is_satisfactory is False, provide 1-3 targeted search queries to find the missing information"
    )
    reasoning: str = Field(description="Brief explanation of the assessment")


async def research_company(state: OverallState, config: RunnableConfig) -> str:
    """Execute a multi-step web search and information extraction process.

    This function performs the following steps:
    1. Generates multiple search queries based on the input query
    2. Executes concurrent web searches using the Tavily API
    3. Deduplicates and formats the search results
    4. Extracts structured information based on the provided schema

    Args:
        query: The initial search query string
        state: Injected application state containing the extraction schema
        config: Runtime configuration for the search process

    Returns:
        str: Structured notes from the search results that are
         relevant to the extraction schema in state.extraction_schema

    Note:
        The function uses concurrent execution for multiple search queries to improve
        performance and combines results from various sources for comprehensive coverage.
    """

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    max_search_queries = configurable.max_search_queries
    max_search_results = configurable.max_search_results

    # Generate search queries
    structured_llm = claude_3_5_sonnet.with_structured_output(Queries)

    # Check reflection output - access attribute directly
    reflection_output = getattr(state, "is_satisfactory", None)
    reflection_queries = getattr(state, "reflection_search_queries", None)

    # If we have performed reflection and have new search queries
    if reflection_output is not None and reflection_queries:
        # Get generated search queries
        query_list = reflection_queries
    else:
        # Format system instructions
        query_instructions = QUERY_WRITER_PROMPT.format(
            company=state.company,
            info=json.dumps(state.extraction_schema, indent=2),
            user_notes=state.user_notes,
            max_search_queries=max_search_queries,
        )

        # Generate queries
        results = structured_llm.invoke(
            [
                {"role": "system", "content": query_instructions},
                {
                    "role": "user",
                    "content": f"Please generate a list of search queries related to the schema that you want to populate.",
                },
            ]
        )

        # Queries
        query_list = [query for query in results.queries]

    # Search tasks
    search_tasks = []
    for query in query_list:
        search_tasks.append(
            tavily_async_client.search(
                query,
                max_results=max_search_results,
                include_raw_content=True,
                topic="general",
            )
        )

    # Execute all searches concurrently
    search_docs = await asyncio.gather(*search_tasks)

    # Deduplicate and format sources
    source_str = deduplicate_and_format_sources(
        search_docs, max_tokens_per_source=1000, include_raw_content=True
    )

    # Generate structured notes relevant to the extraction schema
    p = INFO_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2),
        content=source_str,
        company=state.company,
        user_notes=state.user_notes,
    )
    result = await claude_3_5_sonnet.ainvoke(p)
    return {"completed_notes": [str(result.content)]}


def gather_notes_extract_schema(state: OverallState) -> dict[str, Any]:
    """Gather notes from the web search and extract the schema fields."""

    # Format all notes
    notes = format_all_notes(state.completed_notes)

    # Extract schema fields
    system_prompt = EXTRACTION_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2), notes=notes
    )
    structured_llm = claude_3_5_sonnet.with_structured_output(state.extraction_schema)
    result = structured_llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Produce a structured output from these notes.",
            },
        ]
    )
    return {"info": result}


def reflection(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Reflect on the extracted information and generate search queries to find missing information."""

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Generate search queries
    structured_llm = claude_3_5_sonnet.with_structured_output(ReflectionOutput)

    # Format reflection prompt
    system_prompt = REFLECTION_PROMPT.format(
        schema=json.dumps(state.extraction_schema, indent=2),
        info=state.info,
        max_search_queries=configurable.max_search_queries,
    )

    # Invoke
    result = structured_llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Produce a structured reflection output."},
        ]
    )

    if result.is_satisfactory:
        return {"is_satisfactory": result.is_satisfactory}
    else:
        return {
            "is_satisfactory": result.is_satisfactory,
            "reflection_search_queries": result.reflection_search_queries,
            "reflection_steps_taken": state.reflection_steps_taken + 1,
        }


def route_from_reflection(
    state: OverallState, config: RunnableConfig
) -> Literal[END, "research_company"]:  # type: ignore
    """Route the graph based on the reflection output."""
    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # If we have satisfactory results, end the process
    if state.is_satisfactory:
        return END

    # If results aren't satisfactory but we haven't hit max steps, continue research
    if state.reflection_steps_taken <= configurable.max_reflection_steps:
        return "research_company"

    # If we've exceeded max steps, end even if not satisfactory
    return END


# Add nodes and edges
builder = StateGraph(
    OverallState,
    input=InputState,
    output=OutputState,
    config_schema=Configuration,
)
builder.add_node("gather_notes_extract_schema", gather_notes_extract_schema)
builder.add_node("research_company", research_company)
builder.add_node("reflection", reflection)

builder.add_edge(START, "research_company")
builder.add_edge("research_company", "gather_notes_extract_schema")
builder.add_edge("gather_notes_extract_schema", "reflection")
builder.add_conditional_edges("reflection", route_from_reflection)

# Compile
graph = builder.compile()
