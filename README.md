# Company mAIstro

Company mAIstro researches information about a user-supplied list of companies, and returns it in any user-defined schema.

## Quickstart

1. Populate the `.env` file: 
```
$ cp .env.example .env
```

2. Load this folder in [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio?tab=readme-ov-file#download) 

3. Provide a schema for the output, and pass in a list of companies. 

4. Run the graph! 

## Overview

 Company mAIstro follows a [plan-and-execute workflow](https://github.com/assafelovic/gpt-researcher) that separates planning from research, allowing for better resource management and significantly reducing overall research time:

   - **Planning Phase**: An LLM analyzes the user's set of companies to research and returns a list. 
   - **Research Phase**: The system parallelizes web research across all companies in parallel:
     - Uses [Tavily API](https://tavily.com/) for targeted web searches, performing up to `max_search_queries` queries per company.
     - Performs web searches for each company in parallel and returns up to `max_search_results` results per query.
    - **Extract Schema**: After research is complete, the system uses an LLM to extract the information from the research in the user-defined schema.

## Configuration

The configuration for Company mAIstro is defined in the `configuration.py` file: 
* `max_search_queries`: int = 3 # Max search queries per company
* `max_search_results`: int = 3 # Max search results per query