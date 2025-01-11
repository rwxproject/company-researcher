from dataclasses import dataclass, field
from typing import Any, Optional, Annotated
import operator

from agent.schema import DEFAULT_EXTRACTION_SCHEMA

@dataclass(kw_only=True)
class InputState:
    """Input state defines the interface between the graph and the user (external API)."""

    company: str
    "Company to research provided by the user."

    extraction_schema: dict[str, Any] = field(
        default_factory=lambda: DEFAULT_EXTRACTION_SCHEMA
    )
    "The json schema defines the information the agent is tasked with filling out."

    user_notes: Optional[dict[str, Any]] = field(default=None)
    "Any notes from the user to start the research process."


@dataclass(kw_only=True)
class OverallState:
    """Input state defines the interface between the graph and the user (external API)."""

    company: str
    "Company to research provided by the user."

    extraction_schema: dict[str, Any] = field(
        default_factory=lambda: DEFAULT_EXTRACTION_SCHEMA
    )
    "The json schema defines the information the agent is tasked with filling out."

    user_notes: str = field(default=None)
    "Any notes from the user to start the research process."

    search_queries: list[str] = field(default=None)
    "List of generated search queries to find relevant information"

    completed_notes: Annotated[list, operator.add] = field(default_factory=list)
    "Notes from completed research related to the schema"

    info: dict[str, Any] = field(default=None)
    """
    A dictionary containing the extracted and processed information
    based on the user's query and the graph's execution.
    This is the primary output of the enrichment process.
    """

    is_satisfactory: bool = field(default=None)
    "True if all required fields are well populated, False otherwise"

    reflection_steps_taken: int = field(default=0)
    "Number of times the reflection node has been executed"


@dataclass(kw_only=True)
class OutputState:
    """The response object for the end user.

    This class defines the structure of the output that will be provided
    to the user after the graph's execution is complete.
    """

    info: dict[str, Any]
    """
    A dictionary containing the extracted and processed information
    based on the user's query and the graph's execution.
    This is the primary output of the enrichment process.
    """
