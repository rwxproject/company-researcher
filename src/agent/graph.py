import ast
import asyncio
from typing import cast, Any, Literal, Optional, Annotated
import json
from langchain_core.tools import InjectedToolArg

from langchain_community.tools.bing_search import BingSearchResults
from langchain_community.utilities import BingSearchAPIWrapper
from langchain_anthropic import ChatAnthropic
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field

from agent.configuration import Configuration
from agent.state import InputState, OutputState, OverallState
from agent.utils import deduplicate_and_format_sources, format_all_notes
from agent.prompts import (
    EXTRACTION_PROMPT,
    REFLECTION_PROMPT,
    INFO_PROMPT,
    QUERY_WRITER_PROMPT,
)

# Initialize LLM with rate limiting
rate_limiter = InMemoryRateLimiter(
    requests_per_second=4,
    check_every_n_seconds=0.1,
    max_bucket_size=10,
)
claude_3_5_sonnet = ChatAnthropic(
    model="claude-3-5-sonnet-latest", temperature=0, rate_limiter=rate_limiter
)

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
    search_queries: list[str] = Field(
        description="If is_satisfactory is False, provide 1-3 targeted search queries to find the missing information"
    )
    reasoning: str = Field(description="Brief explanation of the assessment")

async def search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Query Bing search engine.
    
    Returns comprehensive, accurate, and trusted results from the web.
    Particularly useful for current events and company information.
    """
    configuration = Configuration.from_runnable_config(config)
    api_wrapper = BingSearchAPIWrapper(
        k=configuration.max_search_results
    )
    wrapped = BingSearchResults(api_wrapper=api_wrapper)
    result = await wrapped.ainvoke({"query": query})
    return cast(list[dict[str, Any]], result)

def generate_queries(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Generate targeted search queries based on the input and extraction schema."""
    configurable = Configuration.from_runnable_config(config)
    structured_llm = claude_3_5_sonnet.with_structured_output(Queries)

    query_instructions = QUERY_WRITER_PROMPT.format(
        company=state.company,
        info=json.dumps(state.extraction_schema, indent=2),
        user_notes=state.user_notes,
        max_search_queries=configurable.max_search_queries,
    )

    results = cast(
        Queries,
        structured_llm.invoke(
            [
                {"role": "system", "content": query_instructions},
                {
                    "role": "user",
                    "content": "Please generate a list of search queries related to the schema that you want to populate.",
                },
            ]
        ),
    )

    return {"search_queries": results.queries}

async def research_company(
    state: OverallState, config: RunnableConfig
) -> dict[str, Any]:
    """Execute multi-step web search and information extraction process."""
    # Execute parallel search queries
    search_tasks = [search(query, config=config) for query in state.search_queries]
    search_results = await asyncio.gather(*search_tasks)
    
    formatted_docs = []
    for results in search_results:
        if not results:
            continue
            
        # Parse string results if needed
        parsed_results = (
            ast.literal_eval(results) if isinstance(results, str) 
            else results
        )
        
        # Process results list
        if isinstance(parsed_results, list):
            formatted_docs.extend([
                {
                    "url": result.get("link", "Unknown source"),
                    "content": result.get("snippet", ""),
                    "title": result.get("title", "Search Result")
                }
                for result in parsed_results
                if isinstance(result, dict)
            ])

    # Process and format sources
    source_str = deduplicate_and_format_sources(
        formatted_docs, max_tokens_per_source=1000
    )

    # Generate structured notes
    prompt = INFO_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2),
        content=source_str,
        company=state.company,
        user_notes=state.user_notes,
    )
    result = await claude_3_5_sonnet.ainvoke(prompt)
    return {"completed_notes": [str(result.content)]}

def gather_notes_extract_schema(state: OverallState) -> dict[str, Any]:
    """Extract structured information from research notes."""
    notes = format_all_notes(state.completed_notes)
    system_prompt = EXTRACTION_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2), 
        notes=notes
    )
    
    structured_llm = claude_3_5_sonnet.with_structured_output(state.extraction_schema)
    result = structured_llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Produce a structured output from these notes."},
        ]
    )
    return {"info": result}

def reflection(state: OverallState) -> dict[str, Any]:
    """Reflect on extracted information and determine next steps."""
    structured_llm = claude_3_5_sonnet.with_structured_output(ReflectionOutput)

    system_prompt = REFLECTION_PROMPT.format(
        schema=json.dumps(state.extraction_schema, indent=2),
        info=state.info,
    )

    result = cast(
        ReflectionOutput,
        structured_llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Produce a structured reflection output."},
            ]
        ),
    )

    if result.is_satisfactory:
        return {"is_satisfactory": result.is_satisfactory}
    
    return {
        "is_satisfactory": result.is_satisfactory,
        "search_queries": result.search_queries,
        "reflection_steps_taken": state.reflection_steps_taken + 1,
    }

def route_from_reflection(
    state: OverallState, config: RunnableConfig
) -> Literal[END, "research_company"]:
    """Determine next step based on reflection results."""
    configurable = Configuration.from_runnable_config(config)

    if state.is_satisfactory:
        return END

    if state.reflection_steps_taken <= configurable.max_reflection_steps:
        return "research_company"

    return END

# Build and compile graph
builder = StateGraph(
    OverallState,
    input=InputState,
    output=OutputState,
    config_schema=Configuration,
)

builder.add_node("gather_notes_extract_schema", gather_notes_extract_schema)
builder.add_node("generate_queries", generate_queries)
builder.add_node("research_company", research_company)
builder.add_node("reflection", reflection)

# Define graph flow
builder.add_edge(START, "generate_queries")
builder.add_edge("generate_queries", "research_company")
builder.add_edge("research_company", "gather_notes_extract_schema")
builder.add_edge("gather_notes_extract_schema", "reflection")
builder.add_conditional_edges("reflection", route_from_reflection)

# Compile graph
graph = builder.compile()